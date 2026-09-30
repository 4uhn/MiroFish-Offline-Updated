"""Report API routes — generation, retrieval, chat, and agent logs."""

import os
import time
import traceback
import threading
from flask import request, jsonify, send_file, current_app

from . import report_bp
from ..services.report_agent import ReportAgent, ReportManager, ReportStatus, ReportConsoleLogger
from ..services.graph_tools import GraphToolsService
from ..services.simulation_manager import SimulationManager
from ..services.graph_memory_updater import GraphMemoryManager
from ..models.project import ProjectManager
from ..models.task import TaskManager, TaskStatus
from ..utils.logger import get_logger

logger = get_logger('mirofish.api.report')


# ============== Report Generation ==============

@report_bp.route('/generate', methods=['POST'])
def generate_report():
    """Start async report generation task."""
    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return jsonify({"success": False, "error": "simulation_id is required"}), 400

        force_regenerate = data.get('force_regenerate', False)

        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify({"success": False, "error": f"Simulation not found: {simulation_id}"}), 404

        if not force_regenerate:
            existing_report = ReportManager.get_report_by_simulation(simulation_id)
            if existing_report and existing_report.status == ReportStatus.COMPLETED:
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "report_id": existing_report.report_id,
                        "status": "completed",
                        "message": "Report already exists",
                        "already_generated": True
                    }
                })

        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify({"success": False, "error": f"Project not found: {state.project_id}"}), 404

        graph_id = state.graph_id or project.graph_id
        if not graph_id:
            return jsonify({"success": False, "error": "Missing graph_id — ensure graph is built"}), 400

        simulation_requirement = project.simulation_requirement
        if not simulation_requirement:
            return jsonify({"success": False, "error": "Missing simulation requirement"}), 400

        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            return jsonify({"success": False, "error": "Neo4j not connected — check Neo4j service"}), 503
        graph_tools = GraphToolsService(storage=storage, simulation_id=simulation_id)

        import uuid
        report_id = f"report_{uuid.uuid4().hex[:12]}"

        task_manager = TaskManager()
        task_id = task_manager.create_task(
            task_type="report_generate",
            metadata={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "report_id": report_id
            }
        )

        def run_generate():
            try:
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PROCESSING,
                    progress=0,
                    message="Initialising Report Agent..."
                )

                # The report plans from the graph, so wait for the simulation's
                # activity to finish writing to it (Run 8 planned at ~27%).
                drain_timeout = float(os.environ.get('GRAPH_MEMORY_DRAIN_TIMEOUT', '1800'))

                # The report page only shows the report's console log, so write
                # the wait there too (Run 9 sat on a blank page for the drain).
                wait_console = ReportConsoleLogger(report_id)
                wait_log = get_logger('mirofish.report_agent')
                wait_started = time.time()
                first_pending = GraphMemoryManager.pending_count(simulation_id)
                last_pending = None

                def on_wait(pending):
                    nonlocal last_pending
                    task_manager.update_task(
                        task_id,
                        message=f"Waiting for {pending} simulation activities to reach the graph..."
                    )
                    if pending == last_pending:
                        return
                    last_pending = pending
                    done = first_pending - pending
                    eta = ""
                    # Batches of 5 finish 3 at a time, so the rate is noise
                    # until a few batches are done
                    if done >= 10:
                        minutes = (time.time() - wait_started) / done * pending / 60
                        eta = " (<1 min left)" if minutes < 1 else f" (~{minutes:.0f} min left)"
                    wait_log.info(
                        f"Waiting for the simulation to finish writing to the graph: "
                        f"{pending} activities left{eta}"
                    )

                try:
                    drained = GraphMemoryManager.wait_until_drained(simulation_id, drain_timeout, on_wait)
                    if drained:
                        if first_pending:
                            wait_log.info(f"Graph up to date after {time.time() - wait_started:.0f}s, starting report")
                    else:
                        wait_log.warning(
                            f"Graph memory still pending after {drain_timeout:.0f}s "
                            f"({GraphMemoryManager.pending_count(simulation_id)} activities); "
                            f"generating report from a partial graph: simulation_id={simulation_id}"
                        )
                finally:
                    wait_console.close()

                agent = ReportAgent(
                    graph_id=graph_id,
                    simulation_id=simulation_id,
                    simulation_requirement=simulation_requirement,
                    graph_tools=graph_tools
                )

                def progress_callback(stage, progress, message):
                    task_manager.update_task(
                        task_id,
                        progress=progress,
                        message=f"[{stage}] {message}"
                    )

                report = agent.generate_report(
                    progress_callback=progress_callback,
                    report_id=report_id
                )

                ReportManager.save_report(report)

                if report.status == ReportStatus.COMPLETED:
                    task_manager.complete_task(
                        task_id,
                        result={
                            "report_id": report.report_id,
                            "simulation_id": simulation_id,
                            "status": "completed"
                        }
                    )
                else:
                    task_manager.fail_task(task_id, report.error or "Report generation failed")

            except Exception as e:
                logger.error(f"Report generation failed: {e}")
                task_manager.fail_task(task_id, str(e))

        thread = threading.Thread(target=run_generate, daemon=True)
        thread.start()

        return jsonify({
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "report_id": report_id,
                "task_id": task_id,
                "status": "generating",
                "message": "Report generation started — check /api/report/generate/status for progress",
                "already_generated": False
            }
        })

    except Exception as e:
        logger.error(f"Failed to start report generation: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@report_bp.route('/generate/status', methods=['POST'])
def get_generate_status():
    """Query report generation task progress."""
    try:
        data = request.get_json() or {}

        task_id = data.get('task_id')
        simulation_id = data.get('simulation_id')

        if simulation_id:
            existing_report = ReportManager.get_report_by_simulation(simulation_id)
            if existing_report and existing_report.status == ReportStatus.COMPLETED:
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "report_id": existing_report.report_id,
                        "status": "completed",
                        "progress": 100,
                        "message": "Report generated",
                        "already_completed": True
                    }
                })

        if not task_id:
            return jsonify({"success": False, "error": "task_id or simulation_id is required"}), 400

        task_manager = TaskManager()
        task = task_manager.get_task(task_id)

        if not task:
            return jsonify({"success": False, "error": f"Task not found: {task_id}"}), 404

        return jsonify({
            "success": True,
            "data": task.to_dict()
        })

    except Exception as e:
        logger.error(f"Failed to query task status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Report Retrieval ==============

@report_bp.route('/<report_id>', methods=['GET'])
def get_report(report_id: str):
    try:
        report = ReportManager.get_report(report_id)

        if not report:
            return jsonify({"success": False, "error": f"Report not found: {report_id}"}), 404

        return jsonify({
            "success": True,
            "data": report.to_dict()
        })

    except Exception as e:
        logger.error(f"Failed to get report: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@report_bp.route('/by-simulation/<simulation_id>', methods=['GET'])
def get_report_by_simulation(simulation_id: str):
    try:
        report = ReportManager.get_report_by_simulation(simulation_id)

        if not report:
            return jsonify({
                "success": False,
                "error": f"No report for simulation: {simulation_id}",
                "has_report": False
            }), 404

        return jsonify({
            "success": True,
            "data": report.to_dict(),
            "has_report": True
        })

    except Exception as e:
        logger.error(f"Failed to get report: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@report_bp.route('/list', methods=['GET'])
def list_reports():
    try:
        simulation_id = request.args.get('simulation_id')
        limit = request.args.get('limit', 50, type=int)

        reports = ReportManager.list_reports(
            simulation_id=simulation_id,
            limit=limit
        )

        return jsonify({
            "success": True,
            "data": [r.to_dict() for r in reports],
            "count": len(reports)
        })

    except Exception as e:
        logger.error(f"Failed to list reports: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@report_bp.route('/<report_id>/download', methods=['GET'])
def download_report(report_id: str):
    try:
        report = ReportManager.get_report(report_id)

        if not report:
            return jsonify({"success": False, "error": f"Report not found: {report_id}"}), 404

        md_path = ReportManager._get_report_markdown_path(report_id)

        if not os.path.exists(md_path):
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(report.markdown_content)
                temp_path = f.name

            return send_file(
                temp_path,
                as_attachment=True,
                download_name=f"{report_id}.md"
            )

        return send_file(
            md_path,
            as_attachment=True,
            download_name=f"{report_id}.md"
        )

    except Exception as e:
        logger.error(f"Failed to download report: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@report_bp.route('/<report_id>', methods=['DELETE'])
def delete_report(report_id: str):
    try:
        success = ReportManager.delete_report(report_id)

        if not success:
            return jsonify({"success": False, "error": f"Report not found: {report_id}"}), 404

        return jsonify({
            "success": True,
            "message": f"Report deleted: {report_id}"
        })

    except Exception as e:
        logger.error(f"Failed to delete report: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Report Agent Chat ==============

@report_bp.route('/chat', methods=['POST'])
def chat_with_report_agent():
    """Chat with Report Agent — agent can call retrieval tools to answer questions."""
    try:
        data = request.get_json() or {}

        simulation_id = data.get('simulation_id')
        message = data.get('message')
        chat_history = data.get('chat_history', [])

        if not simulation_id:
            return jsonify({"success": False, "error": "simulation_id is required"}), 400

        if not message:
            return jsonify({"success": False, "error": "message is required"}), 400

        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify({"success": False, "error": f"Simulation not found: {simulation_id}"}), 404

        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify({"success": False, "error": f"Project not found: {state.project_id}"}), 404

        graph_id = state.graph_id or project.graph_id
        if not graph_id:
            return jsonify({"success": False, "error": "Missing graph_id"}), 400

        simulation_requirement = project.simulation_requirement or ""

        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            return jsonify({"success": False, "error": "Neo4j not connected — check Neo4j service"}), 503
        graph_tools = GraphToolsService(storage=storage, simulation_id=simulation_id)

        agent = ReportAgent(
            graph_id=graph_id,
            simulation_id=simulation_id,
            simulation_requirement=simulation_requirement,
            graph_tools=graph_tools
        )

        result = agent.chat(message=message, chat_history=chat_history)

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Report Progress & Sections ==============

@report_bp.route('/<report_id>/progress', methods=['GET'])
def get_report_progress(report_id: str):
    try:
        progress = ReportManager.get_progress(report_id)

        if not progress:
            return jsonify({"success": False, "error": f"Report not found or no progress: {report_id}"}), 404

        return jsonify({
            "success": True,
            "data": progress
        })

    except Exception as e:
        logger.error(f"Failed to get report progress: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@report_bp.route('/<report_id>/sections', methods=['GET'])
def get_report_sections(report_id: str):
    """Get generated sections (for incremental streaming to frontend)."""
    try:
        sections = ReportManager.get_generated_sections(report_id)

        report = ReportManager.get_report(report_id)
        is_complete = report is not None and report.status == ReportStatus.COMPLETED

        return jsonify({
            "success": True,
            "data": {
                "report_id": report_id,
                "sections": sections,
                "total_sections": len(sections),
                "is_complete": is_complete
            }
        })

    except Exception as e:
        logger.error(f"Failed to get sections: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@report_bp.route('/<report_id>/section/<int:section_index>', methods=['GET'])
def get_single_section(report_id: str, section_index: int):
    try:
        section_path = ReportManager._get_section_path(report_id, section_index)

        if not os.path.exists(section_path):
            return jsonify({"success": False, "error": f"Section not found: section_{section_index:02d}.md"}), 404

        with open(section_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return jsonify({
            "success": True,
            "data": {
                "filename": f"section_{section_index:02d}.md",
                "section_index": section_index,
                "content": content
            }
        })

    except Exception as e:
        logger.error(f"Failed to get section: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Report Status Check ==============

@report_bp.route('/check/<simulation_id>', methods=['GET'])
def check_report_status(simulation_id: str):
    try:
        report = ReportManager.get_report_by_simulation(simulation_id)

        has_report = report is not None
        report_status = report.status.value if report else None
        report_id = report.report_id if report else None
        interview_unlocked = has_report and report.status == ReportStatus.COMPLETED

        return jsonify({
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "has_report": has_report,
                "report_status": report_status,
                "report_id": report_id,
                "interview_unlocked": interview_unlocked
            }
        })

    except Exception as e:
        logger.error(f"Failed to check report status: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Agent Log ==============

@report_bp.route('/<report_id>/agent-log', methods=['GET'])
def get_agent_log(report_id: str):
    """Get structured agent execution log (incremental via from_line param)."""
    try:
        from_line = request.args.get('from_line', 0, type=int)
        log_data = ReportManager.get_agent_log(report_id, from_line=from_line)

        return jsonify({
            "success": True,
            "data": log_data
        })

    except Exception as e:
        logger.error(f"Failed to get agent log: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@report_bp.route('/<report_id>/agent-log/stream', methods=['GET'])
def stream_agent_log(report_id: str):
    try:
        logs = ReportManager.get_agent_log_stream(report_id)

        return jsonify({
            "success": True,
            "data": {
                "logs": logs,
                "count": len(logs)
            }
        })

    except Exception as e:
        logger.error(f"Failed to get agent log: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Console Log ==============

@report_bp.route('/<report_id>/console-log', methods=['GET'])
def get_console_log(report_id: str):
    """Get plain-text console output log (incremental via from_line param)."""
    try:
        from_line = request.args.get('from_line', 0, type=int)
        log_data = ReportManager.get_console_log(report_id, from_line=from_line)

        return jsonify({
            "success": True,
            "data": log_data
        })

    except Exception as e:
        logger.error(f"Failed to get console log: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@report_bp.route('/<report_id>/console-log/stream', methods=['GET'])
def stream_console_log(report_id: str):
    try:
        logs = ReportManager.get_console_log_stream(report_id)

        return jsonify({
            "success": True,
            "data": {
                "logs": logs,
                "count": len(logs)
            }
        })

    except Exception as e:
        logger.error(f"Failed to get console log: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Debug Tools ==============

@report_bp.route('/tools/search', methods=['POST'])
def search_graph_tool():
    try:
        data = request.get_json() or {}

        graph_id = data.get('graph_id')
        query = data.get('query')
        limit = data.get('limit', 10)

        if not graph_id or not query:
            return jsonify({"success": False, "error": "graph_id and query are required"}), 400

        from ..services.graph_tools import GraphToolsService
        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage not initialized — check Neo4j connection")
        tools = GraphToolsService(storage=storage)
        result = tools.search_graph(graph_id=graph_id, query=query, limit=limit)

        return jsonify({
            "success": True,
            "data": result.to_dict()
        })

    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@report_bp.route('/tools/statistics', methods=['POST'])
def get_graph_statistics_tool():
    try:
        data = request.get_json() or {}

        graph_id = data.get('graph_id')

        if not graph_id:
            return jsonify({"success": False, "error": "graph_id is required"}), 400

        from ..services.graph_tools import GraphToolsService
        storage = current_app.extensions.get('neo4j_storage')
        if not storage:
            raise ValueError("GraphStorage not initialized — check Neo4j connection")
        tools = GraphToolsService(storage=storage)
        result = tools.get_graph_statistics(graph_id)

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as e:
        logger.error(f"Failed to get graph statistics: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
