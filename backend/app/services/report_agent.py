"""
Report Agent Service
Generates simulation reports using the ReACT pattern (via GraphStorage / Neo4j)

Features:
1. Generate reports based on simulation requirements and graph data
2. Plan the outline first, then generate section by section
3. Each section uses ReACT multi-round reasoning and reflection
4. Supports conversational interaction with autonomous tool invocation
"""

import os
import json
import re
from collections import Counter
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .graph_tools import (
    GraphToolsService
)

logger = get_logger('mirofish.report_agent')


class ReportLogger:
    """
    Report Agent Detailed Logger

    Generates an agent_log.jsonl file in the report folder, recording each step in detail.
    Each line is a complete JSON object containing timestamp, action type, details, etc.
    """

    def __init__(self, report_id: str):
        """
        Initialize the logger

        Args:
            report_id: Report ID, used to determine the log file path
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'agent_log.jsonl'
        )
        self.start_time = datetime.now()
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Ensure the log file directory exists"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _get_elapsed_time(self) -> float:
        """Get elapsed time from start in seconds"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def log(
        self, 
        action: str, 
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None
    ):
        """
        Record a log entry

        Args:
            action: Action type, e.g. 'start', 'tool_call', 'llm_response', 'section_complete', etc.
            stage: Current stage, e.g. 'planning', 'generating', 'completed'
            details: Details dictionary, not truncated
            section_title: Current section title (optional)
            section_index: Current section index (optional)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details
        }
        
        # Append to JSONL file
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """Record report generation start"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": "Report generation task started"
            }
        )

    def log_planning_start(self):
        """Record outline planning start"""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": "Starting report outline planning"}
        )

    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """Record outline planning completion"""
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": "Outline planning complete",
                "outline": outline_dict
            }
        )

    def log_section_start(self, section_title: str, section_index: int):
        """Record section generation start"""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": f"Starting section generation: {section_title}"}
        )

    def log_tool_call(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        parameters: Dict[str, Any],
        iteration: int
    ):
        """Record tool invocation"""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": f"Calling tool: {tool_name}"
            }
        )

    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int
    ):
        """Record tool invocation result (full content, not truncated)"""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # Full result, not truncated
                "result_length": len(result),
                "message": f"Tool {tool_name} returned result"
            }
        )
    
    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool
    ):
        """Record LLM response (full content, not truncated)"""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # Full response, not truncated
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": f"LLM response (tool calls: {has_tool_calls}, final answer: {has_final_answer})"
            }
        )
    
    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int
    ):
        """Record section content generation (content only, does not mean the section is fully complete)"""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # Full content, not truncated
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": f"Section {section_title} content generation complete"
            }
        )
    
    def log_section_full_complete(
        self,
        section_title: str,
        section_index: int,
        full_content: str
    ):
        """
        Record section generation completion

        The frontend should listen for this log to determine if a section is truly complete and get the full content
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": f"Section {section_title} generation complete"
            }
        )
    
    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """Record report generation completion"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": "Report generation complete"
            }
        )

    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """Record an error"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": f"Error occurred: {error_message}"
            }
        )


class ReportConsoleLogger:
    """
    Report Agent Console Logger

    Writes console-style logs (INFO, WARNING, etc.) to a console_log.txt file in the report folder.
    These logs differ from agent_log.jsonl and are plain-text console output.
    """

    def __init__(self, report_id: str):
        """
        Initialize the console logger

        Args:
            report_id: Report ID, used to determine the log file path
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'console_log.txt'
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()
    
    def _ensure_log_file(self):
        """Ensure the log file directory exists"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _setup_file_handler(self):
        """Set up the file handler to write logs to file simultaneously"""
        import logging
        
        # Create file handler
        self._file_handler = logging.FileHandler(
            self.log_file_path,
            mode='a',
            encoding='utf-8'
        )
        self._file_handler.setLevel(logging.INFO)
        
        # Use the same concise format as the console
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self._file_handler.setFormatter(formatter)
        
        # Attach to report_agent related loggers
        loggers_to_attach = [
            'mirofish.report_agent',
            'mirofish.graph_tools',
        ]

        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # Avoid duplicate handlers
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)
    
    def close(self):
        """Close the file handler and remove it from loggers"""
        import logging
        
        if self._file_handler:
            loggers_to_detach = [
                'mirofish.report_agent',
                'mirofish.graph_tools',
            ]

            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)
            
            self._file_handler.close()
            self._file_handler = None
    
    def __del__(self):
        """Ensure file handler is closed on destruction"""
        self.close()


class ReportStatus(str, Enum):
    """Report status"""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """Report section"""
    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content
        }

    def to_markdown(self, level: int = 2) -> str:
        """Convert to Markdown format"""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Report outline"""
    title: str
    summary: str
    sections: List[ReportSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def to_markdown(self) -> str:
        """Convert to Markdown format"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """Complete report"""
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


# ═══════════════════════════════════════════════════════════════
# Prompt Template Constants
# ═══════════════════════════════════════════════════════════════

# ── Tool Descriptions ──

TOOL_DESC_INSIGHT_FORGE = """\
[Deep Insight Retrieval - Powerful retrieval tool]
This is our powerful retrieval function, designed for in-depth analysis. It will:
1. Automatically decompose your question into multiple sub-questions
2. Retrieve information from the simulation graph across multiple dimensions
3. Integrate results from semantic search, entity analysis, and relationship chain tracing
4. Return the most comprehensive and in-depth retrieval content

[Use Cases]
- Need to deeply analyze a topic
- Need to understand multiple aspects of an event
- Need to gather rich material to support report sections

[Returns]
- Extracted facts (machine summaries: paraphrase, never quote)
- Core entity insights
- Relationship chain analysis
- What agents actually wrote on the topic (verbatim posts/comments: quotable)"""

TOOL_DESC_PANORAMA_SEARCH = """\
[Broad Search - Get full panoramic view]
This tool retrieves the complete overview of simulation results, especially suitable for understanding event evolution. It will:
1. Retrieve all relevant nodes and relationships
2. Distinguish between currently valid facts and historical/expired facts
3. Help you understand how public opinion has evolved

[Use Cases]
- Need to understand the complete development trajectory of events
- Need to compare opinion changes across different stages
- Need comprehensive entity and relationship information

[Returns]
- Currently valid facts (machine summaries: paraphrase, never quote)
- Historical/expired facts (evolution records)
- All involved entities
- What agents actually wrote on the topic (verbatim posts/comments: quotable)"""

TOOL_DESC_QUICK_SEARCH = """\
[Simple Search - Quick retrieval]
A lightweight, fast retrieval tool suitable for simple, direct information queries.

[Use Cases]
- Need to quickly look up specific information
- Need to verify a fact
- Simple information retrieval

[Returns]
- List of facts most relevant to the query (machine summaries: paraphrase, never quote)
- What agents actually wrote on the topic (verbatim posts/comments: quotable)"""

TOOL_DESC_SIMULATION_STATS = """\
[Simulation Statistics - counts of what agents wrote]
Counts the posts, quotes and comments agents wrote: totals, members of the public
versus institutions, the most active authors, and for each keyword how many texts
and agents mention it.

[Use Cases]
- Before writing that a view spread, gained traction, dominated or was rare: check how many agents wrote about it
- Need the size of the discussion or who drove it

[Returns]
- Counts only (nothing quotable)"""

TOOL_DESC_INTERVIEW_AGENTS = """\
[In-depth Interview - Real Agent Interview (dual platform)]
Calls the OASIS simulation environment's interview API to conduct real interviews with running simulation Agents!
This is not an LLM simulation — it calls real interview endpoints to get original answers from simulation Agents.
By default, interviews are conducted simultaneously on both Twitter and Reddit platforms for more comprehensive perspectives.

Workflow:
1. Automatically reads persona files to understand all simulation Agents
2. Selects a mix of members of the public and institutions, favouring agents not yet interviewed
3. Automatically generates questions that every interviewee answers
4. Calls the /api/simulation/interview/batch endpoint to conduct real interviews on both platforms

[Use Cases]
- Need to understand event perspectives from different roles (What do students think? What does the media say? What is the official stance?)
- Need to collect opinions and positions from multiple parties
- Need to get real answers from simulation Agents (from the OASIS simulation environment)
- Want to make the report more vivid with "interview transcripts"

[Returns]
- Identity information of interviewed Agents
- Each Agent's interview responses on both Twitter and Reddit platforms
- Key quotes (can be quoted directly)

[Important] The OASIS simulation environment must be running to use this feature!"""

# ── Outline Planning Prompt ──

PLAN_SYSTEM_PROMPT = """\
You are an expert in writing "Future Prediction Reports," with a "god's eye view" of the simulation world — you can observe every Agent's behavior, statements, and interactions within the simulation.

Respond in English only.

[Core Concept]
We have built a simulation world and injected specific "simulation requirements" as variables. The evolution of the simulation world represents a prediction of what could happen in the future. What you are observing is not "experimental data" but a "rehearsal of the future."

[Your Task]
Write a "Future Prediction Report" that answers:
1. Under the conditions we set, what happened in the future?
2. How did the various Agents (groups of people) react and act?
3. What noteworthy future trends and risks does this simulation reveal?

[Report Positioning]
- This is a simulation-based future prediction report, revealing "if this happens, what would the future look like"
- Focus on prediction results: event trajectories, group reactions, emergent phenomena, potential risks
- The behavior and statements of Agents in the simulation world are predictions of future human behavior
- This is NOT an analysis of the current real world
- This is NOT a generic opinion summary

[Titles and Scope - the outline is written before any evidence is gathered]
- Each section title names a question or topic the report will examine, never its answer. The requirement asks "whether the warnings gain traction": the title is "Response to the Campaigners' 2019 Warnings", not "Campaigners' Warnings Gain Traction". Not "Anger Surges", not "Anxiety Outpaces Reassurance".
- The summary says in one sentence what the report examines. It states no findings: those are written after the sections.
- The measured counts below show what agents actually wrote about. A topic with 0 texts still gets examined if the requirement asks about it, under a neutral title.

[Section Count Limits]
- Minimum 2 sections, maximum 5 sections
- No sub-sections needed — each section should contain complete content
- Content should be concise and focused on core predictive findings
- You design the section structure based on the prediction results

Output a JSON-formatted report outline in the following format:
{
    "title": "Report Title",
    "summary": "One sentence saying what the report examines (no findings)",
    "sections": [
        {
            "title": "Section title naming a question, not an answer",
            "description": "What the section will examine"
        }
    ]
}

Note: The sections array must have at least 2 and at most 5 elements!"""

# Constrains outline decoding: json_object mode alone let qwen3 emit
# section objects without a "title" key, producing untitled sections.
OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "sections": {
            "type": "array",
            "minItems": 2,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["title", "description"],
            },
        },
    },
    "required": ["title", "summary", "sections"],
}

PLAN_USER_PROMPT_TEMPLATE = """\
[Prediction Scenario]
The variable injected into the simulation world (simulation requirement): {simulation_requirement}

[Simulation World Scale]
- Number of entities in the simulation: {total_nodes}
- Number of relationships between entities: {total_edges}
- Entity type distribution: {entity_types}
- Number of active Agents: {total_entities}

[Sample Future Facts Predicted by the Simulation]
{related_facts_json}

{measured_counts}
Examine this future rehearsal from a "god's eye view":
1. Under the conditions we set, what state did the future present?
2. How did different groups (Agents) react and act?
3. What noteworthy future trends does this simulation reveal?

Design the most appropriate report section structure based on the prediction results.

[Reminder] Number of report sections: minimum 2, maximum 5. Content should be concise and focused on core predictive findings."""

# ── Section Generation Prompt ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert in writing "Future Prediction Reports," and you are currently writing one section of the report.

Respond in English only. Write the report in English.

Report Title: {report_title}
Report scope: {report_summary}
Prediction Scenario (Simulation Requirement): {simulation_requirement}

{measured_counts}
Current section to write: {section_title}

==========
[Core Concept]
==========

The simulation world is a rehearsal of the future. We injected specific conditions (simulation requirements) into the simulation world. The behavior and interactions of Agents in the simulation are predictions of future human behavior.

Your task is to:
- Reveal what happened in the future under the set conditions
- Predict how various groups (Agents) reacted and acted
- Identify noteworthy future trends, risks, and opportunities

Do NOT write this as an analysis of the current real world.
Focus on "what the future would look like" — the simulation results ARE the predicted future.

==========
[Most Important Rules - Must Follow]
==========

1. [You MUST call tools to observe the simulation world]
   - You are observing the future rehearsal from a "god's eye view"
   - All content must come from events and Agent behavior in the simulation world
   - Do NOT use your own knowledge to write report content
   - Each section must call tools at least 3 times (maximum 5) to observe the simulated world, which represents the future

2. [Quote only what agents actually said]
   - Two things are quotable: interview answers, and entries under "What agents actually wrote". Copy them word for word and name the speaker.
   - Everything else in tool results (facts, entity summaries, relationship chains) is a machine-written summary. Paraphrase it in your own sentences; never put it in quotation marks or a > block quote.
   - A fact marked [agent claim] was extracted from what simulated agents posted, not from the source document. Report it as something agents said or believed ("several residents claimed..."), never as an established fact.
   - An interview answer is that agent's own account. Report what it says the agent did or believes as its claim ("Persimmon Homes told interviewers it had..."), not as something that happened, and credit its quotes with "interview" in place of a platform. A handful of interviewees is not a measure of how common a view was.
   - Use at most 6 block quotes per section, each a different speaker or point, and never repeat a quote already used in a completed section. Weave the rest into prose.
   - Quote any one speaker at most 2 times in the whole report; the checker removes further quotes from them. Give the space to other voices.
   - Block quote format, for example:
     > "<exact words from an interview answer or agent post>" (Speaker name, role, platform or "interview")
   - Every quote is checked against the tool output after you finish; a quote that is not word for word is turned into plain text

3. [Language Consistency - All content must be in English]
   - Tool results may contain content in various languages
   - When quoting tool results in other languages, you must translate them into fluent English before including them in the report
   - Preserve the original meaning while ensuring natural, smooth expression
   - This rule applies to both body text and block quotes (> format)

4. [STRICTLY ground all content in tool results — ZERO fabrication]
   - Report content must reflect ONLY what the tools returned. Every claim, quote, and statistic MUST trace back to a specific tool result from this session.
   - NEVER fabricate quotes, statistics, percentages, platform names, organization names, or events that were not in tool results.
   - NEVER invent social media platforms (e.g., Facebook, Instagram, TikTok) or channels unless they explicitly appeared in tool results.
   - NEVER create fictional quotes from agents who did not appear in tool results. Only quote agents whose exact words were returned by tools.
   - If a tool returns limited or no relevant data for a topic, say so honestly: "The simulation data on this aspect was limited" — do NOT fill gaps with plausible-sounding invented content.
   - Claims of spread, reach or change ("gained traction", "spread rapidly", "widely shared", "many residents", "trust fluctuated", "growing distrust") need a measured basis: cite the count (e.g. "3 of 80 texts, by 3 agents") from the measured counts or simulation_stats, or a comparison of what agents wrote early and late in the run. Without one, report what the named agents said and state that the data is thin.
   - When quoting an agent, use ONLY their actual words from tool results. Do not paraphrase loosely or embellish.
   - Do not start a quote or sentence with the speaker announcing their role ("As a journalist, ..."); cut to the substance with "..." if needed.
   - If you are unsure whether something came from a tool result, DO NOT include it.
   - Write about the simulated world, never about your research: do not mention tools, searches, search terms, "the texts" or "the data". When the measured counts show agents did not discuss a topic, say plainly that they did not, and do not then suggest it was referenced or implied.
   - The section title names the question this section answers, not the answer. Let the evidence decide what you conclude, even when it is "agents barely discussed this".

==========
[Format Requirements - Extremely Important!]
==========

[One section = smallest content unit]
- Each section is the smallest unit of the report
- Do NOT use any Markdown headings (#, ##, ###, #### etc.) within sections
- Do NOT add the section title at the beginning of the content
- Section titles are added automatically by the system — you only need to write the body text
- Organise content with paragraph breaks, block quotes, lists and a few bolded key phrases inside sentences, but do NOT use headings

[Correct Example]
```
<opening sentence answering the section's question, with a count from the measured counts where one applies>

**<key phrase>** <sentence about what named agents wrote or told interviewers>:

> "<exact words from an agent post>" (Speaker name, role, platform)

<sentence on a second view, from a different agent>:

- <point supported by a tool result>
- <another point supported by a tool result>
```

[Incorrect Example]
```
## Executive Summary          <- Wrong! Do not add any headings
### 1. Initial Phase          <- Wrong! Do not use ### for sub-sections
#### 1.1 Detailed Analysis    <- Wrong! Do not use #### for subdivisions

This section analyzes...
```

==========
[Available Retrieval Tools] (call 3-5 times per section)
==========

{tools_description}

[Tool Usage Tips - Mix different tools, do not use only one type]
- insight_forge: Deep insight analysis, automatically decomposes questions and retrieves facts and relationships from multiple dimensions
- panorama_search: Wide-angle panoramic search, understand the full picture, timeline, and evolution of events
- quick_search: Quickly verify a specific data point
- interview_agents: Interview simulation Agents, get first-person perspectives and real reactions from different roles
- simulation_stats: Count how many agents wrote about something; check this before saying a view spread, gained traction or dominated

==========
[Workflow]
==========

Each reply you can only do ONE of the following two things (not both):

Option A - Call a tool:
Output your reasoning, then call a tool using the following format:
<tool_call>
{{"name": "tool_name", "parameters": {{"param_name": "param_value"}}}}
</tool_call>
The system will execute the tool and return the results to you. You do not need to and cannot write tool results yourself.

Option B - Output final content:
When you have gathered enough information through tools, output the section content starting with "Final Answer:"

Strictly prohibited:
- Do NOT include both a tool call and Final Answer in a single reply
- Do NOT fabricate tool results (Observations) — all tool results are injected by the system
- Call at most one tool per reply

==========
[Section Content Requirements]
==========

1. Content must be based on simulation data retrieved via tools
2. Support points with agents' own words (interview answers and agent posts only, at most 6 block quotes), and paraphrase extracted facts
3. Use Markdown formatting (but no headings):
   - Bold a few key phrases inside sentences (instead of sub-headings); never write a sentence about the formatting itself
   - Use lists (- or 1. 2. 3.) to organize points
   - Use blank lines to separate paragraphs
   - Do NOT use #, ##, ###, #### or any heading syntax
4. [Block Quote Format - Must be standalone paragraphs]
   Quotes must be standalone paragraphs with a blank line before and after — do not embed them within a paragraph:

   Correct format:
   ```
   <sentence introducing the quote>

   > "<exact words from an interview answer or agent post>" (Speaker name, role, platform or "interview")

   <sentence following it up>
   ```

   Incorrect format:
   ```
   The response was considered lacking. > "The response appeared rigid..." This assessment reflects...
   ```
5. Maintain logical coherence with other sections
6. [Avoid Repetition] Carefully read the completed sections below and do not repeat the same information
7. [Emphasis] Do NOT add any headings! Bold key phrases inside sentences instead of sub-headings"""

SECTION_USER_PROMPT_TEMPLATE = """\
Completed sections (read carefully to avoid repetition):
{previous_content}

==========
[Current Task] Write section: {section_title}
==========

[Important Reminders]
1. Carefully read the completed sections above to avoid repeating the same content!
2. You must call tools to retrieve simulation data before writing
3. Mix different tools — do not use only one type
4. Report content must come from retrieval results — do not use your own knowledge

[Format Warning - Must Follow]
- Do NOT write any headings (#, ##, ###, #### are all prohibited)
- Do NOT write "{section_title}" as the opening line
- Section titles are added automatically by the system
- Write body text directly, bolding key phrases inside sentences instead of sub-headings

Begin:
1. First, think (Thought) about what information this section needs
2. Then call tools (Action) to retrieve simulation data
3. Once you have enough information, output Final Answer (body text only, no headings)"""

# ── ReACT Loop Message Templates ──

REACT_OBSERVATION_TEMPLATE = """\
Observation (retrieval results):

=== Tool {tool_name} returned ===
{result}

==========
Tools called {tool_calls_count}/{max_tool_calls} times (used: {used_tools_str}){unused_hint}
- If you have enough information: output section content starting with "Final Answer:" (quote only interview answers and agent posts above, word for word)
- If you need more information: call a tool to continue retrieval
=========="""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "[Notice] You have only called tools {tool_calls_count} times — at least {min_tool_calls} calls are required. "
    "Please call more tools to retrieve additional simulation data before outputting Final Answer. {unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "Currently only {tool_calls_count} tool calls have been made — at least {min_tool_calls} are required. "
    "Please call tools to retrieve simulation data. {unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "Tool call limit reached ({tool_calls_count}/{max_tool_calls}) — no more tool calls allowed. "
    'Please immediately output the section content starting with "Final Answer:" based on the information already retrieved.'
)

REACT_UNUSED_TOOLS_HINT = "\nTip: You have not used: {unused_list} yet. Consider trying different tools to get multi-perspective information."

REACT_FORCE_FINAL_MSG = "Tool call limit reached. Please output Final Answer: and generate the section content now."

# ── Chat Prompt ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
You are a concise and efficient simulation prediction assistant.

Respond in English only.

[Background]
Prediction conditions: {simulation_requirement}

[Generated Analysis Report]
{report_content}

[Rules]
1. Prioritize answering based on the report content above
2. Answer questions directly — avoid lengthy reasoning
3. Only call tools to retrieve more data when the report content is insufficient
4. Answers should be concise, clear, and well-organized

[Available Tools] (use only when needed, maximum 1-2 calls)
{tools_description}

[Tool Call Format]
<tool_call>
{{"name": "tool_name", "parameters": {{"param_name": "param_value"}}}}
</tool_call>

[Response Style]
- Be concise and direct — avoid lengthy exposition
- Use > format to quote key content
- Lead with conclusions, then explain reasoning"""

CHAT_OBSERVATION_SUFFIX = "\n\nPlease answer the question concisely."


# ═══════════════════════════════════════════════════════════════
# ReportAgent Main Class
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent - Simulation Report Generation Agent

    Uses the ReACT (Reasoning + Acting) pattern:
    1. Planning phase: Analyze simulation requirements, plan report outline structure
    2. Generation phase: Generate content section by section, each section can call tools multiple times
    3. Reflection phase: Check content completeness and accuracy
    """

    # Maximum tool calls per section
    MAX_TOOL_CALLS_PER_SECTION = 5

    # Maximum tool calls per chat turn
    MAX_TOOL_CALLS_PER_CHAT = 2

    # Tokens kept free for each section-writing reply. Run 12's longest
    # section was 2,957 chars (~750 tokens).
    SECTION_REPLY_TOKENS = 1536
    # Chars per token until the server reports a prompt's token count
    # (deliberately low: a low guess trims more, never overflows)
    DEFAULT_CHARS_PER_TOKEN = 3.0
    # Highest chars/token taken from a measurement. When a chat exceeds
    # num_ctx, Ollama drops whole older messages (logged only at debug level)
    # and reports the tokens of what was left, so an overfull prompt measures
    # implausibly sparse: Run 13 section 4 read ~5.8, with its task gone.
    # Measured on qwen3: the section system prompt, the densest part, is 4.68
    # (it was 4.4 with its 63-char '═' rules at 32 tokens each); trimmed
    # section prompts, mostly system prompt, 4.70-4.81.
    MAX_CHARS_PER_TOKEN = 4.6
    # A measurement above this means messages were almost certainly dropped
    # (interview transcripts, the sparsest part measured, are 5.5)
    DROPPED_MESSAGES_CHARS_PER_TOKEN = 5.6
    # Share of the prompt budget used, for content that tokenizes denser
    # than the prompt the chars-per-token figure was measured on
    CONTEXT_SAFETY = 0.95
    
    def __init__(
        self,
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        graph_tools: Optional[GraphToolsService] = None
    ):
        """
        Initialize Report Agent

        Args:
            graph_id: Graph ID
            simulation_id: Simulation ID
            simulation_requirement: Simulation requirement description
            llm_client: LLM client (optional)
            graph_tools: Graph tools service (optional, requires externally injected GraphStorage)
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        # Words agents actually said that this section's tools returned
        # (interview answers, agent posts); the only text a section may quote
        self._section_quotable: List[str] = []
        # (speaker, text) for each quotable text, to check who a quote is credited to
        self._section_speakers: List[Tuple[str, str]] = []
        self.simulation_requirement = simulation_requirement

        self.llm = llm_client or LLMClient()
        # Measured from the server's prompt_tokens after each section call
        self._chars_per_token = self.DEFAULT_CHARS_PER_TOKEN
        if graph_tools is None:
            raise ValueError(
                "graph_tools (GraphToolsService) is required. "
                "Create it via GraphToolsService(storage=...) and pass it in."
            )
        if graph_tools.simulation_id != simulation_id:
            # An unscoped service would mix other runs' agent claims into this report
            raise ValueError(
                f"graph_tools is scoped to {graph_tools.simulation_id!r}, not {simulation_id!r}. "
                "Create it via GraphToolsService(storage=..., simulation_id=...)."
            )
        self.graph_tools = graph_tools
        
        # Tool definitions
        self.tools = self._define_tools()
        # Mention counts for what the requirement asks about (_measure_requirement)
        self._measured_counts = ""
        # Requirement keywords no agent text contains

        # Logger (initialized in generate_report)
        self.report_logger: Optional[ReportLogger] = None
        # Console logger (initialized in generate_report)
        self.console_logger: Optional[ReportConsoleLogger] = None

        logger.info(f"ReportAgent initialized: graph_id={graph_id}, simulation_id={simulation_id}")
    
    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Define available tools"""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "The question or topic you want to analyze in depth",
                    "report_context": "Context of the current report section (optional, helps generate more precise sub-questions)"
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Search query, used for relevance ranking",
                    "include_expired": "Whether to include expired/historical content (default True)"
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "Search query string",
                    "limit": "Number of results to return (optional, default 10)"
                }
            },
            "simulation_stats": {
                "name": "simulation_stats",
                "description": TOOL_DESC_SIMULATION_STATS,
                "parameters": {
                    "keywords": "Comma-separated words or phrases to count mentions of (optional, up to 10)"
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "Interview topic or requirement description (e.g., 'Understand students\\'  views on the dormitory formaldehyde incident')",
                    "max_agents": "Maximum number of Agents to interview (optional, default 5, max 10)"
                }
            }
        }
    
    MAX_REQUIREMENT_KEYWORDS = 10

    def _measure_requirement(self) -> str:
        """Mention counts for the topics the simulation requirement asks about.

        Run 11's report said the campaigners' 2019 warnings "were referenced
        in discussions"; no agent text mentioned 2019, a warning or a
        petition. The writer had the simulation_stats tool but never asked it.
        These counts are measured once and given to every section, so the
        report's answers to the requirement's own questions rest on them.
        """
        from .agent_posts import AgentPostIndex
        prompt = f"""Simulation requirement:
{self.simulation_requirement}

List up to {self.MAX_REQUIREMENT_KEYWORDS} search terms that a social media post would contain if it discussed one of the things this requirement asks to observe. Cover every question or topic it names, including organisations and past events.
- Each term is ONE word, or one organisation's name. A term matches words that start with it, so use the shortest form: "warn" matches warned and warnings, "apolog" matches apology and apologise.
- Use words a post would actually contain ("2019", "petition", "apolog", "Persimmon"), never labels or combinations ("public sentiment", "Persimmon blame").

Return JSON: {{"keywords": ["...", "..."]}}"""
        try:
            result = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": "You pick search terms. Respond in English only. Return pure JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            keywords = [k.strip() for k in result.get("keywords", []) if isinstance(k, str) and k.strip()]
        except Exception as e:
            logger.warning(f"Could not pick requirement keywords, sections get no measured counts: {e}")
            return ""
        keywords = list(dict.fromkeys(keywords))[:self.MAX_REQUIREMENT_KEYWORDS]
        if not keywords:
            logger.warning("No requirement keywords returned, sections get no measured counts")
            return ""
        stats = AgentPostIndex.for_simulation(self.simulation_id).stats_text(keywords)
        logger.info(f"Measured requirement keywords: {keywords}")
        return (
            "Measured in the simulation (counts of what agents wrote, for the questions above). "
            "Words in 0 texts were never used by an agent: say so plainly, and never describe them as "
            "mentioned, referenced or gaining traction. Cite a count with its own unit "
            "(\"8 texts by 5 agents, 3 of them public agents\"), never \"3 public texts\".\n" + stats + "\n"
        )

    AGENT_POSTS_PER_SEARCH = 6

    def _agent_posts_block(self, query: str) -> str:
        """Verbatim agent posts/comments matching the query, appended to search tool output.

        Graph facts are NER paraphrases; these are the words agents wrote, and
        with interview answers they are the only text a section may quote.
        """
        from .agent_posts import AgentPostIndex
        try:
            posts = AgentPostIndex.for_simulation(self.simulation_id).search(query, k=self.AGENT_POSTS_PER_SEARCH)
        except Exception as e:
            logger.warning(f"Agent post lookup failed for {self.simulation_id}: {e}")
            posts = []
        if not posts:
            return "\n\n### What agents actually wrote\n(No agent posts matched this query.)"
        self._section_quotable.extend(p.content for p in posts)
        self._section_speakers.extend((p.name, p.content) for p in posts)
        lines = [
            "\n\n### What agents actually wrote "
            "(verbatim posts/comments: quote word for word and name the speaker)"
        ]
        lines += [f"{i}. {p.to_text()}" for i, p in enumerate(posts, 1)]
        return "\n".join(lines)

    _QUOTE_OPEN = '"\u201c'
    _QUOTE_CLOSE = '"\u201d'

    _INLINE_QUOTE = re.compile(r'["“]([^"“”\n]+)["”]')

    @staticmethod
    def _sentences(line: str) -> List[str]:
        """A prose line's sentences, with their leading spaces; never split inside a quote."""
        out, start, inside, n = [], 0, False, len(line)
        for i, ch in enumerate(line):
            # "... the crisis.** While" ends after the bold marker
            j = i + 1
            while j < n and line[j] == '*':
                j += 1
            # 'wrote: "... failed to pro..." indicating a lack' goes on
            k = j
            while k < n and line[k].isspace():
                k += 1
            ends = (j == n or line[j].isspace()) and not (k < n and line[k].islower())
            if ch == '“' or (ch == '"' and not inside):
                inside = True
            elif ch == '”' or (ch == '"' and inside):
                inside = False
                # tweeted: "... in their blood?" This sentiment ...
                if ends and i > 0 and line[i - 1] in '.!?':
                    out.append(line[start:j])
                    start = j
            elif not inside and ch in '.!?' and ends:
                out.append(line[start:j])
                start = j
        out.append(line[start:])
        return [s for s in out if s.strip()]

    @classmethod
    def _drop_unverified_inline(cls, text: str, corpus: str) -> Tuple[str, int]:
        """Remove each sentence with an inline quote of four or more words
        that is not word for word in corpus (normalised agent texts).

        Until Run 13 such a quote only lost its quotation marks, so "As one
        resident put it, Trust is hard to rebuild once it's broken" stayed in
        the report: words no agent wrote, still credited to a resident.
        """
        norm = GraphToolsService._normalise_for_quote_match

        def verified(quoted: str) -> bool:
            parts = [p for p in re.split(r'\.\.\.|…', quoted) if p.strip()]
            return len(quoted.split()) < 4 or all(norm(p) in corpus for p in parts)

        dropped = 0
        lines = []
        for line in text.split("\n"):
            sentences = cls._sentences(line)
            kept = [s for s in sentences
                    if all(verified(m.group(1)) for m in cls._INLINE_QUOTE.finditer(s))]
            if len(kept) < len(sentences):
                dropped += len(sentences) - len(kept)
                line = "".join(kept).strip()
            lines.append(line)
        return "\n".join(lines), dropped

    @classmethod
    def _verify_quotes(cls, content: str, sources: List[str],
                       speakers: Optional[List[Tuple[str, str]]] = None) -> tuple:
        """Keep only quotes that are agents' own words, credited to their author.

        In Run 9 a section block-quoted ~40 graph facts (NER paraphrases such
        as "Kwame Patel responded ... by demanding accountability") as if
        agents had said them. A block quote is kept only if its text (each
        part, when elided with "...") is word for word in an interview answer
        or agent post this section's tools returned; otherwise it is removed.
        A sentence with an inline "..." quote that fails the same check is
        removed too (_drop_unverified_inline).

        speakers lists (name, text) for the sources. Run 11 credited Oliver
        Taylor's post, word for word, to Tom Patel. When a block quote's
        attribution names one of the speakers, one of the texts containing
        the quote must be theirs, or the quote is removed.
        """
        norm = GraphToolsService._normalise_for_quote_match
        corpus = " \n ".join(norm(s) for s in sources)
        stats = {"kept": 0, "dropped": 0, "misattributed": 0, "inline_dropped": 0}
        # "Drinking Water Inspectorate (DWI)" is also credited without the acronym
        speaker_texts = [
            (re.sub(r"\s*\([^)]*\)", "", name).strip().lower(), norm(text))
            for name, text in (speakers or []) if name
        ]
        known_names = {name for name, _ in speaker_texts if name}

        def misattributed(quoted: str, credit: str) -> bool:
            credit = credit.lower()
            named = {n for n in known_names if re.search(rf"\b{re.escape(n)}\b", credit)}
            parts = [norm(p) for p in re.split(r'\.\.\.|\u2026', quoted) if len(p.split()) >= 3]
            authors = {n for n, t in speaker_texts if all(p in t for p in parts)}
            return bool(named) and not (named & authors)

        def verbatim(text: str) -> bool:
            parts = [p for p in re.split(r'\.\.\.|\u2026', text) if len(p.split()) >= 3]
            return bool(parts) and all(norm(p) in corpus for p in parts)

        def split_quote(body: str) -> tuple:
            """(lead-in, quoted text, trailing attribution) of a block quote body.

            Handles `"text" (Name)` and `**Name:** "text"`; a body without
            quotation marks is treated as all quoted text.
            """
            body = body.strip()
            starts = [k for k in (body.find(c) for c in cls._QUOTE_OPEN) if k >= 0]
            if starts:
                start = min(starts)
                end = max(body.rfind(c) for c in cls._QUOTE_CLOSE)
                if end > start:
                    return body[:start].strip(), body[start + 1:end].strip(), body[end + 1:].strip()
            return "", body, ""

        out: List[str] = []
        prose: List[str] = []

        def flush_prose():
            if prose:
                text, n = cls._drop_unverified_inline("\n".join(prose), corpus)
                stats["inline_dropped"] += n
                out.append(text)
                prose.clear()

        for line in content.split("\n"):
            if not line.lstrip().startswith(">"):
                prose.append(line)
                continue
            # Each > line is checked on its own: the model often stacks several
            # quotes, from different sources, in one block
            flush_prose()
            body = line.lstrip()[1:].strip()
            lead, quoted, attribution = split_quote(body)
            if len(quoted.split()) < 4:
                out.append(f"> {body}" if body else ">")
            elif not verbatim(quoted):
                stats["dropped"] += 1
            elif misattributed(quoted, f"{lead} {attribution}"):
                stats["misattributed"] += 1
            else:
                stats["kept"] += 1
                out.append(f"\n> {body}\n")
        flush_prose()
        content = "\n".join(out)
        return re.sub(r'\n{3,}', '\n\n', content), stats

    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        """
        Execute a tool call

        Args:
            tool_name: Tool name
            parameters: Tool parameters
            report_context: Report context (used for InsightForge)

        Returns:
            Tool execution result (text format)
        """
        # Normalize degenerate param_name/param_value format from LLM
        if "param_name" in parameters and "param_value" in parameters:
            real_key = parameters.pop("param_name")
            real_val = parameters.pop("param_value")
            parameters[real_key] = real_val

        logger.info(f"Executing tool: {tool_name}, parameters: {parameters}")

        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "") or ""
                if not query.strip():
                    query = self.simulation_requirement
                    logger.info(f"Empty query for insight_forge, using simulation_requirement: {query[:60]}")
                ctx = parameters.get("report_context", "") or report_context
                result = self.graph_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx
                )
                return result.to_text() + self._agent_posts_block(query)

            elif tool_name == "panorama_search":
                query = parameters.get("query", "") or ""
                if not query.strip():
                    query = self.simulation_requirement
                    logger.info(f"Empty query for panorama_search, using simulation_requirement: {query[:60]}")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ['true', '1', 'yes']
                result = self.graph_tools.panorama_search(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired
                )
                return result.to_text() + self._agent_posts_block(query)

            elif tool_name == "quick_search":
                query = parameters.get("query", "") or ""
                if not query.strip():
                    query = self.simulation_requirement
                    logger.info(f"Empty query for quick_search, using simulation_requirement: {query[:60]}")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.graph_tools.quick_search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text() + self._agent_posts_block(query)
            
            elif tool_name == "interview_agents":
                # In-depth interview - call real OASIS interview API to get simulation Agent responses (dual platform)
                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.graph_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents
                )
                # Failed interviews carry a bracketed placeholder, not agent words
                answered = [i for i in result.interviews
                            if not re.match(r'\[(?!Twitter\]|Reddit\])', i.response)]
                self._section_quotable.extend(i.response for i in answered)
                self._section_speakers.extend((i.agent_name, i.response) for i in answered)
                return result.to_text()

            elif tool_name == "simulation_stats":
                from .agent_posts import AgentPostIndex
                keywords = parameters.get("keywords", "")
                if isinstance(keywords, str):
                    keywords = keywords.split(",")
                return AgentPostIndex.for_simulation(self.simulation_id).stats_text(keywords)

            # ========== Backward-compatible legacy tools (internally redirected to new tools) ==========
            
            elif tool_name == "search_graph":
                # Redirect to quick_search
                logger.info("search_graph redirected to quick_search")
                return self._execute_tool("quick_search", parameters, report_context)
            
            elif tool_name == "get_graph_statistics":
                result = self.graph_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.graph_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_simulation_context":
                # Redirect to insight_forge since it is more powerful
                logger.info("get_simulation_context redirected to insight_forge")
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool("insight_forge", {"query": query}, report_context)
            
            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.graph_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return f"Unknown tool: {tool_name}. Please use one of: {', '.join(sorted(self.VALID_TOOL_NAMES))}"
                
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}, error: {str(e)}")
            return f"Tool execution failed: {str(e)}"
    
    # Valid tool name set, used for validation during bare JSON fallback parsing
    VALID_TOOL_NAMES = {"insight_forge", "panorama_search", "quick_search", "interview_agents", "simulation_stats"}

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from LLM response

        Supported formats (by priority):
        1. <tool_call>{"name": "tool_name", "parameters": {...}}</tool_call>
        2. Bare JSON (response body or single line is a tool call JSON)
        """
        tool_calls = []

        # Format 1: XML style (standard format)
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Format 2: Fallback - LLM outputs bare JSON (without <tool_call> tags)
        # Only attempted when format 1 doesn't match, to avoid false matches on JSON in body text
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # Response may contain reasoning text + bare JSON; try to extract the last JSON object
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """Validate whether the parsed JSON is a valid tool call"""
        # Support both {"name": ..., "parameters": ...} and {"tool": ..., "params": ...} key formats
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # Normalize key names to name / parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False
    
    def _get_tools_description(self) -> str:
        """Generate tool description text"""
        desc_parts = ["Available tools:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  Parameters: {params_desc}")
        return "\n".join(desc_parts)
    
    def plan_outline(
        self,
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        Plan report outline

        Use LLM to analyze simulation requirements and plan the report structure

        Args:
            progress_callback: Progress callback function

        Returns:
            ReportOutline: Report outline
        """
        logger.info("Starting report outline planning...")
        
        if progress_callback:
            progress_callback("planning", 0, "Analyzing simulation requirements...")

        # First, get simulation context
        context = self.graph_tools.get_simulation_context(
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement
        )
        
        if progress_callback:
            progress_callback("planning", 30, "Generating report outline...")
        
        system_prompt = PLAN_SYSTEM_PROMPT
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
            measured_counts=self._measured_counts,
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                schema=OUTLINE_SCHEMA,
                schema_name="report_outline",
            )
            
            if progress_callback:
                progress_callback("planning", 80, "Parsing outline structure...")
            
            # Parse outline
            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content=""
                ))
            
            outline = ReportOutline(
                title=response.get("title", "Simulation Analysis Report"),
                summary=response.get("summary", ""),
                sections=sections
            )
            
            if progress_callback:
                progress_callback("planning", 100, "Outline planning complete")

            logger.info(f"Outline planning complete: {len(sections)} sections")
            return outline
            
        except Exception as e:
            logger.error(f"Outline planning failed: {str(e)}")
            # Return default outline (3 sections, as fallback)
            return ReportOutline(
                title="Future Prediction Report",
                summary="Future trends and risk analysis based on simulation predictions",
                sections=[
                    ReportSection(title="Prediction Scenario and Core Findings"),
                    ReportSection(title="Population Behavior Prediction Analysis"),
                    ReportSection(title="Trend Outlook and Risk Alerts")
                ]
            )
    
    _LIST_ITEM = re.compile(r"^\s*(?:\d+\.|-)\s")
    _OMITTED = re.compile(r"^\((\d+) more omitted to fit the context window\)$")

    @classmethod
    def _drop_list_items(cls, text: str, need: int) -> Tuple[str, int]:
        """Remove items from the end of the longest list in text, until `need` chars are gone.

        One item of each list is kept, and a note says how many were
        omitted. Returns the new text and the chars removed (0 when no list
        has an item to spare).
        """
        lines = text.split("\n")
        blocks, start = [], None
        for i in range(len(lines) + 1):
            if i < len(lines) and cls._LIST_ITEM.match(lines[i]):
                if start is None:
                    start = i
            elif start is not None:
                if i - start > 1:
                    blocks.append((start, i))
                start = None
        if not blocks:
            return text, 0
        first, end = max(blocks, key=lambda b: b[1] - b[0])
        # A list trimmed before already ends in a note; fold the counts together
        note = cls._OMITTED.match(lines[end]) if end < len(lines) else None
        omitted = int(note.group(1)) if note else 0
        tail_end = end + 1 if note else end
        cut = end
        removed = 0
        while cut > first + 1 and removed < need:
            cut -= 1
            removed += len(lines[cut]) + 1
        omitted += end - cut
        lines[cut:tail_end] = [f"({omitted} more omitted to fit the context window)"]
        new_text = "\n".join(lines)
        return new_text, len(text) - len(new_text)

    _INTERVIEW = re.compile(r"(?=\n#### Interview #\d+: )")
    _INTERVIEWS_OMITTED = re.compile(r"\n\((\d+) more interviews omitted to fit the context window\)\s*$")
    _SECTION_OMITTED = "(rest of this section omitted to fit the context window)"
    PREVIOUS_SECTIONS_SEP = "\n\n---\n\n"

    @classmethod
    def _drop_interviews(cls, text: str, need: int) -> Tuple[str, int]:
        """Remove whole interviews from the end of an interview_agents result.

        Interviewees are listed in selection order, so the least relevant go
        first. One interview is always kept, and a note says how many were
        omitted. Returns the new text and the chars removed.
        """
        note = cls._INTERVIEWS_OMITTED.search(text)
        omitted = int(note.group(1)) if note else 0
        body = text[:note.start()] if note else text
        blocks = cls._INTERVIEW.split(body)
        # blocks[0] is the header; each later block is one interview
        if len(blocks) <= 2:
            return text, 0
        removed = 0
        while len(blocks) > 2 and removed < need:
            removed += len(blocks.pop())
            omitted += 1
        new_text = "".join(blocks) + f"\n({omitted} more interviews omitted to fit the context window)"
        return new_text, len(text) - len(new_text)

    @classmethod
    def _drop_paragraphs(cls, text: str, need: int) -> Tuple[str, int]:
        """Shorten the longest completed section in the previous-sections text.

        Paragraphs go from the end; each section keeps its heading and first
        paragraph. Returns the new text and the chars removed.
        """
        sections = [sec.split("\n\n") for sec in text.split(cls.PREVIOUS_SECTIONS_SEP)]
        # Sections shortened before end in a note; set it aside while counting
        noted = {i for i, paras in enumerate(sections) if paras[-1] == cls._SECTION_OMITTED}
        for i in noted:
            sections[i].pop()
        candidates = [i for i, paras in enumerate(sections) if len(paras) > 2]
        if not candidates:
            return text, 0
        longest = max(candidates, key=lambda i: sum(len(x) for x in sections[i]))
        paras = sections[longest]
        removed = 0
        while len(paras) > 2 and removed < need:
            removed += len(paras.pop()) + 2
        for i in noted | {longest}:
            sections[i].append(cls._SECTION_OMITTED)
        new_text = cls.PREVIOUS_SECTIONS_SEP.join("\n\n".join(ps) for ps in sections)
        return new_text, len(text) - len(new_text)

    def _fit_context(self, messages: List[Dict[str, str]], trimmable: List[Dict[str, Any]],
                     section_title: str) -> None:
        """Shrink the prompt until it leaves SECTION_REPLY_TOKENS of the window free.

        Run 12 sent two section prompts of ~8,040 tokens into the 8,192-token
        window, and Run 13 three of 7,600-7,900 (with interview transcripts).
        Ollama does not reject that: it shifts the context mid-reply, dropping
        the start of the prompt (the system prompt with the quoting, framing
        and overclaiming rules) while the section is still being written. A
        chat already over the window before the reply is worse: Ollama drops
        whole messages after the system prompt, the section task first
        (Run 13 sections 3 and 4).

        trimmable lists the prompt parts that can shrink: {"kind", "index"
        (message), "text", "render" (text -> message content)}. The completed
        sections go first (they are there only to avoid repetition), then the largest tool
        result: its longest list (InsightForge's machine-extracted Key Facts),
        then whole interviews. Verbatim agent posts are never trimmed.
        """
        budget = int((self.llm.num_ctx - self.SECTION_REPLY_TOKENS)
                     * self._chars_per_token * self.CONTEXT_SAFETY)
        size = sum(len(m["content"]) for m in messages)
        trimmed = Counter()
        while size > budget:
            need = size - budget
            order = [t for t in trimmable if t["kind"] == "previous"] + sorted(
                (t for t in trimmable if t["kind"] == "tool"), key=lambda t: len(t["text"]), reverse=True)
            for item in order:
                shrinkers = [self._drop_paragraphs] if item["kind"] == "previous" \
                    else [self._drop_list_items, self._drop_interviews]
                for shrink in shrinkers:
                    new_text, removed = shrink(item["text"], need)
                    if removed > 0:
                        break
                if removed > 0:
                    break
            else:
                logger.error(
                    f"Section {section_title}: prompt is {size} chars, over the {budget}-char budget "
                    f"({self.llm.num_ctx} tokens minus {self.SECTION_REPLY_TOKENS} for the reply), "
                    "and nothing is left to trim"
                )
                return
            item["text"] = new_text
            old_len = len(messages[item["index"]]["content"])
            messages[item["index"]]["content"] = item["render"](new_text)
            size += len(messages[item["index"]]["content"]) - old_len
            trimmed[item["kind"]] += old_len - len(messages[item["index"]]["content"])
        if trimmed:
            logger.info(f"Section {section_title}: trimmed {trimmed.get('previous', 0)} chars of completed "
                        f"sections and {trimmed.get('tool', 0)} chars of tool results to fit "
                        f"{self.llm.num_ctx} tokens ({size} chars at {self._chars_per_token:.2f} chars/token)")

    def _section_llm_call(self, messages: List[Dict[str, str]], trimmable: List[Dict[str, Any]],
                          section_title: str) -> str:
        """One section-writing LLM call, fitted to the context window first."""
        self._fit_context(messages, trimmable, section_title)
        sent = sum(len(m["content"]) for m in messages)
        result = self.llm.chat_result(
            messages=messages,
            temperature=0.5,
            max_tokens=self.SECTION_REPLY_TOKENS,
        )
        if result.prompt_tokens:
            measured = sent / result.prompt_tokens
            if measured > self.DROPPED_MESSAGES_CHARS_PER_TOKEN:
                logger.warning(f"Section {section_title}: {sent} chars counted as {result.prompt_tokens} tokens "
                               f"({measured:.2f} chars/token); Ollama probably dropped earlier messages "
                               f"to fit {self.llm.num_ctx} tokens")
            self._chars_per_token = min(measured, self.MAX_CHARS_PER_TOKEN)
            logger.info(f"Section {section_title}: prompt {result.prompt_tokens} tokens "
                        f"({measured:.2f} chars/token), reply {result.completion_tokens} tokens")
        if result.finish_reason == "length":
            logger.warning(f"Section {section_title}: reply reached max_tokens={self.SECTION_REPLY_TOKENS} and was cut off")
        return result.content

    SUMMARY_CHARS_PER_SECTION = 3000

    def _write_summary(self, outline: ReportOutline, sections: List[str]) -> str:
        """One-sentence report summary, written after the sections from what they say.

        Until Run 13 the planner wrote it before any tool call, from the
        requirement and ten graph facts ("a surge in health anxiety, while
        campaigners' warnings gain traction"), and every section writer was
        given it as the report's findings. The planner now writes a scope
        line; this replaces it.
        """
        body = "\n\n".join(sec[:self.SUMMARY_CHARS_PER_SECTION] for sec in sections)
        prompt = f"""Simulation requirement:
{self.simulation_requirement}

{self._measured_counts}
The report's sections:
{body}

Write the one sentence (at most 45 words) that opens this report, summarising what its sections found. Use only claims the sections make, keeping their hedges and counts. Do not add claims of spread, growth, polarisation or traction ("surge", "widespread", "gained traction") that no section supports with a count. Return only the sentence."""
        try:
            summary = self.llm.chat(
                messages=[
                    {"role": "system", "content": "You write report summaries. Respond in English only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=200,
            ).strip().strip('"').strip()
        except Exception as e:
            logger.warning(f"Could not write the report summary; keeping the planned scope line: {e}")
            return outline.summary
        if not summary:
            logger.warning("Report summary came back empty; keeping the planned scope line")
            return outline.summary
        logger.info(f"Report summary: {summary}")
        return summary

    def _generate_section_react(
        self,
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        """
        Generate a single section using the ReACT pattern

        ReACT loop:
        1. Thought - Analyze what information is needed
        2. Action - Call tools to retrieve information
        3. Observation - Analyze tool results
        4. Repeat until sufficient information or max iterations reached
        5. Final Answer - Generate section content

        Args:
            section: The section to generate
            outline: Complete outline
            previous_sections: Content of previous sections (for coherence)
            progress_callback: Progress callback
            section_index: Section index (for logging)

        Returns:
            Section content (Markdown format)
        """
        logger.info(f"ReACT generating section: {section.title}")
        
        # Log section start
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        
        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            measured_counts=self._measured_counts,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )

        # Build user prompt - each completed section limited to 4000 characters
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # Each section limited to 4000 characters
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = self.PREVIOUS_SECTIONS_SEP.join(previous_parts)
        else:
            previous_content = "(This is the first section)"

        def render_user_prompt(previous: str) -> str:
            return SECTION_USER_PROMPT_TEMPLATE.format(previous_content=previous, section_title=section.title)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": render_user_prompt(previous_content)}
        ]
        # Prompt parts _fit_context may shrink: completed sections, then tool results
        trimmable: List[Dict[str, Any]] = []
        if previous_sections:
            trimmable.append({"kind": "previous", "index": 1, "text": previous_content,
                              "render": render_user_prompt})
        
        # ReACT loop
        tool_calls_count = 0
        max_iterations = 7  # Maximum iteration rounds (increased from 5 for local models)
        min_tool_calls = 2  # Minimum tool calls required (reduced from 3 for reliability)
        conflict_retries = 0  # Consecutive conflict count when tool call and Final Answer appear together
        used_tools = set()  # Track which tools have been called
        all_tools = set(self.VALID_TOOL_NAMES)

        # Report context, used for InsightForge sub-question generation
        report_context = f"Section title: {section.title}\nSimulation requirement: {self.simulation_requirement}"
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating", 
                    int((iteration / max_iterations) * 100),
                    f"Deep retrieval and writing ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})"
                )
            
            # Call LLM
            response = self._section_llm_call(messages, trimmable, section.title)

            # Check if LLM returned None (API error or empty content)
            if response is None:
                logger.warning(f"Section {section.title} iteration {iteration + 1}: LLM returned None")
                # If there are remaining iterations, add message and retry
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(empty response)"})
                    messages.append({"role": "user", "content": "Please continue generating content."})
                    continue
                # Last iteration also returned None, break to forced conclusion
                break

            logger.debug(f"LLM response: {response[:200]}...")

            # Parse once and reuse results
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── Conflict handling: LLM output both tool call and Final Answer ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    f"Section {section.title} iteration {iteration+1}: "
                    f"LLM output both tool call and Final Answer (conflict #{conflict_retries})"
                )

                if conflict_retries <= 2:
                    # First two times: discard this response and ask LLM to reply again
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "[Format Error] Your reply contains both a tool call and Final Answer, which is not allowed.\n"
                            "Each reply can only do one of the following:\n"
                            "- Call a tool (output a <tool_call> block, do NOT write Final Answer)\n"
                            "- Output final content (start with 'Final Answer:', do NOT include <tool_call>)\n"
                            "Please reply again, doing only one of these."
                        ),
                    })
                    continue
                else:
                    # Third time: degrade by truncating to the first tool call and forcing execution
                    logger.warning(
                        f"Section {section.title}: {conflict_retries} consecutive conflicts, "
                        "degrading to truncated execution of first tool call"
                    )
                    first_tool_end = response.find('</tool_call>')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call>')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # Log LLM response
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )

            # ── Case 1: LLM output Final Answer ──
            if has_final_answer:
                # Insufficient tool calls, reject and require more tool calls
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = f"(These tools have not been used yet, consider trying them: {', '.join(unused_tools)})" if unused_tools else ""
                    messages.append({
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    })
                    continue

                # Normal completion
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(f"Section {section.title} generation complete (tool calls: {tool_calls_count})")

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count
                    )
                return final_answer

            # ── Case 2: LLM attempted to call a tool ──
            if has_tool_calls:
                # Tool quota exhausted -> explicitly inform, require Final Answer
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": REACT_TOOL_LIMIT_MSG.format(
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        ),
                    })
                    continue

                # Only execute the first tool call
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(f"LLM attempted to call {len(tool_calls)} tools, executing only the first: {call['name']}")

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                used_tools.add(call['name'])

                # Build unused tools hint
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(unused_list=", ".join(unused_tools))

                obs_kwargs = dict(
                    tool_name=call["name"],
                    tool_calls_count=tool_calls_count,
                    max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                    used_tools_str=", ".join(used_tools),
                    unused_hint=unused_hint,
                )
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": REACT_OBSERVATION_TEMPLATE.format(result=result, **obs_kwargs),
                })
                trimmable.append({
                    "kind": "tool", "index": len(messages) - 1, "text": result,
                    "render": lambda r, kw=obs_kwargs: REACT_OBSERVATION_TEMPLATE.format(result=r, **kw),
                })
                continue

            # ── Case 3: Neither tool call nor Final Answer ──
            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                # Insufficient tool calls, recommend unused tools
                unused_tools = all_tools - used_tools
                unused_hint = f"(These tools have not been used yet, consider trying them: {', '.join(unused_tools)})" if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # Sufficient tool calls made, LLM output content without "Final Answer:" prefix
            # Directly accept this content as the final answer
            logger.info(f"Section {section.title}: 'Final Answer:' prefix not detected, adopting LLM output as final content (tool calls: {tool_calls_count})")
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer
        
        # Max iterations reached, force content generation
        logger.warning(f"Section {section.title} reached max iterations, forcing generation")
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})
        
        response = self._section_llm_call(messages, trimmable, section.title)

        # Check if LLM returned None during forced conclusion
        if response is None:
            logger.error(f"Section {section.title}: LLM returned None during forced conclusion, using default error message")
            final_answer = f"(This section could not be generated. Please try again with a different model or configuration.)"
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response

        # Strip any leaked tool_call markup from forced generation
        final_answer = ReportManager._strip_tool_call_markup(final_answer)

        # If final answer is empty or trivially short after cleanup, add a placeholder
        if not final_answer or len(final_answer.strip()) < 20:
            logger.warning(f"Section {section.title}: forced generation produced empty/trivial content")
            final_answer = f"(This section could not be fully generated due to insufficient data from the simulation. The report agent exhausted its iteration budget before producing content.)"
        
        # Log section content generation completion
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count
            )
        
        return final_answer
    
    def generate_report(
        self,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        """
        Generate complete report (real-time section-by-section output)

        Each section is saved to the folder immediately after generation, no need to wait for the entire report.
        File structure:
        reports/{report_id}/
            meta.json       - Report metadata
            outline.json    - Report outline
            progress.json   - Generation progress
            section_01.md   - Section 1
            section_02.md   - Section 2
            ...
            full_report.md  - Complete report

        Args:
            progress_callback: Progress callback function (stage, progress, message)
            report_id: Report ID (optional, auto-generated if not provided)

        Returns:
            Report: Complete report
        """
        import uuid
        
        # Auto-generate report_id if not provided
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        # List of completed section titles (for progress tracking)
        completed_section_titles = []
        
        try:
            # Initialize: create report folder and save initial state
            ReportManager._ensure_report_folder(report_id)
            
            # Initialize logger (structured log agent_log.jsonl)
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )
            
            # Initialize console logger (console_log.txt)
            self.console_logger = ReportConsoleLogger(report_id)
            
            ReportManager.update_progress(
                report_id, "pending", 0, "Initializing report...",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            # Phase 1: Plan outline
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, "Starting report outline planning...",
                completed_sections=[]
            )
            
            # Log planning start
            self.report_logger.log_planning_start()
            
            if progress_callback:
                progress_callback("planning", 0, "Starting report outline planning...")
            
            # Measured before planning: the outline and every section use them
            self._measured_counts = self._measure_requirement()

            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: 
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline
            
            # Log planning completion
            self.report_logger.log_planning_complete(outline.to_dict())
            
            # Save outline to file
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, f"Outline planning complete, {len(outline.sections)} sections",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            logger.info(f"Outline saved to file: {report_id}/outline.json")

            # Phase 2: Generate section by section (save each section)
            report.status = ReportStatus.GENERATING
            
            total_sections = len(outline.sections)
            generated_sections = []  # Save content for context
            
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)
                
                # Update progress
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    f"Generating section: {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )
                
                if progress_callback:
                    progress_callback(
                        "generating", 
                        base_progress, 
                        f"Generating section: {section.title} ({section_num}/{total_sections})"
                    )
                
                # Generate main section content
                self._section_quotable = []
                self._section_speakers = []
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage, 
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None,
                    section_index=section_num
                )
                
                # Quotes must be agents' own words from this section's tools
                section_content, qstats = self._verify_quotes(
                    section_content, self._section_quotable, speakers=self._section_speakers,
                )
                logger.info(
                    f"Section {section.title}: quotes kept={qstats['kept']}, "
                    f"dropped={qstats['dropped']} (not verbatim agent text), "
                    f"misattributed={qstats['misattributed']}, "
                    f"inline_dropped={qstats['inline_dropped']} (sentences quoting words no agent wrote)"
                )

                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # Save section
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Log section completion
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(f"Section saved: {report_id}/section_{section_num:02d}.md")
                
                # Update progress
                ReportManager.update_progress(
                    report_id, "generating",
                    base_progress + int(70 / total_sections),
                    f"Section {section.title} complete",
                    current_section=None,
                    completed_sections=completed_section_titles
                )
            
            # The summary opening the report is written from what the sections found
            outline.summary = self._write_summary(outline, generated_sections)
            ReportManager.save_outline(report_id, outline)

            # Phase 3: Assemble complete report
            if progress_callback:
                progress_callback("generating", 95, "Assembling complete report...")
            
            ReportManager.update_progress(
                report_id, "generating", 95, "Assembling complete report...",
                completed_sections=completed_section_titles
            )
            
            # Use ReportManager to assemble the complete report
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()
            
            # Calculate total time
            total_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Log report completion
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )
            
            # Save final report
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, "Report generation complete",
                completed_sections=completed_section_titles
            )
            
            if progress_callback:
                progress_callback("completed", 100, "Report generation complete")
            
            logger.info(f"Report generation complete: {report_id}")

            # Close console logger
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            report.status = ReportStatus.FAILED
            report.error = str(e)
            
            # Log error
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")
            
            # Save failure state
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, f"Report generation failed: {str(e)}",
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # Ignore save failure errors
            
            # Close console logger
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None

            return report

    def chat(
        self,
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Chat with the Report Agent

        During conversation, the Agent can autonomously call retrieval tools to answer questions

        Args:
            message: User message
            chat_history: Chat history

        Returns:
            {
                "response": "Agent reply",
                "tool_calls": [list of tools called],
                "sources": [information sources]
            }
        """
        logger.info(f"Report Agent chat: {message[:50]}...")

        # Quotable text is per section/turn; left over from report generation
        # it grows without bound and admits another section's sources
        self._section_quotable = []
        self._section_speakers = []
        chat_history = chat_history or []
        
        # Get generated report content
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # Limit report length to avoid context overflow
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [Report content truncated] ..."
        except Exception as e:
            logger.warning(f"Failed to get report content: {e}")
        
        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "(No report available yet)",
            tools_description=self._get_tools_description(),
        )

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history
        for h in chat_history[-10:]:  # Limit history length
            messages.append(h)
        
        # Add user message
        messages.append({
            "role": "user", 
            "content": message
        })
        
        # ReACT loop (simplified)
        tool_calls_made = []
        max_iterations = 2  # Reduced iteration rounds
        
        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5
            )
            
            # Parse tool calls
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # No tool calls, return response directly
                clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
                
                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }
            
            # Execute tool calls (limited count)
            tool_results = []
            for call in tool_calls[:1]:  # Max 1 tool call per iteration
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]  # Limit result length
                })
                tool_calls_made.append(call)
            
            # Add results to messages
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join([f"[{r['tool']} result]\n{r['result']}" for r in tool_results])
            messages.append({
                "role": "user",
                "content": observation + CHAT_OBSERVATION_SUFFIX
            })
        
        # Max iterations reached, get final response
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )
        
        # Clean response
        clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL)
        clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
        
        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:
    """
    Report Manager

    Responsible for persistent storage and retrieval of reports

    File structure (section-by-section output):
    reports/
      {report_id}/
        meta.json          - Report metadata and status
        outline.json       - Report outline
        progress.json      - Generation progress
        section_01.md      - Section 1
        section_02.md      - Section 2
        ...
        full_report.md     - Complete report
    """

    # Report storage directory
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')
    
    @classmethod
    def _ensure_reports_dir(cls):
        """Ensure the reports root directory exists"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
    
    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Get report folder path"""
        return os.path.join(cls.REPORTS_DIR, report_id)
    
    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """Ensure report folder exists and return path"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Get report metadata file path"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")
    
    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """Get full report Markdown file path"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")
    
    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Get outline file path"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")
    
    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Get progress file path"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")
    
    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """Get section Markdown file path"""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")
    
    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Get Agent log file path"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")
    
    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Get console log file path"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")
    
    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Get console log content

        These are console output logs (INFO, WARNING, etc.) during report generation,
        different from the structured agent_log.jsonl logs.

        Args:
            report_id: Report ID
            from_line: Line number to start reading from (for incremental retrieval, 0 means from start)

        Returns:
            {
                "logs": [list of log lines],
                "total_lines": total line count,
                "from_line": starting line number,
                "has_more": whether there are more logs
            }
        """
        log_path = cls._get_console_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # Keep original log line, strip trailing newline
                    logs.append(line.rstrip('\n\r'))
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Read to end
        }

    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        Get complete console log (one-time full retrieval)

        Args:
            report_id: Report ID

        Returns:
            List of log lines
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Get Agent log content

        Args:
            report_id: Report ID
            from_line: Line number to start reading from (for incremental retrieval, 0 means from start)

        Returns:
            {
                "logs": [list of log entries],
                "total_lines": total line count,
                "from_line": starting line number,
                "has_more": whether there are more logs
            }
        """
        log_path = cls._get_agent_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # Skip lines that fail to parse
                        continue
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Read to end
        }

    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Get complete Agent log (one-time full retrieval)

        Args:
            report_id: Report ID

        Returns:
            List of log entries
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        Save report outline

        Called immediately after planning phase completes
        """
        cls._ensure_report_folder(report_id)
        
        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Outline saved: {report_id}")
    
    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        """
        Save a single section

        Called immediately after each section is generated, enabling section-by-section output

        Args:
            report_id: Report ID
            section_index: Section index (starting from 1)
            section: Section object

        Returns:
            Saved file path
        """
        cls._ensure_report_folder(report_id)

        # Build section Markdown content - clean up possible duplicate headings
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Save file
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"Section saved: {report_id}/{file_suffix}")
        return file_path
    
    @classmethod
    def _strip_tool_call_markup(cls, content: str) -> str:
        """
        Remove any raw <tool_call> or ```tool_call markup that leaked into generated content.
        """
        import re
        # Remove <tool_call>...</tool_call> blocks
        content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)
        # Remove unclosed <tool_call> blocks (at end of content)
        content = re.sub(r'<tool_call>.*$', '', content, flags=re.DOTALL)
        # Remove ```tool_call...``` blocks
        content = re.sub(r'```tool_call.*?```', '', content, flags=re.DOTALL)
        # Remove lines that are just tool call JSON (common LLM leak pattern)
        content = re.sub(r'^\s*\{"name":\s*"(insight_forge|panorama_search|quick_search|interview_agents|simulation_stats)".*$', '', content, flags=re.MULTILINE)
        # Clean up excessive blank lines left behind
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content.strip()

    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        Clean section content

        1. Strip leaked tool_call markup
        2. Remove Markdown heading lines at the beginning that duplicate the section title
        3. Convert all ### and lower level headings to bold text

        Args:
            content: Original content
            section_title: Section title

        Returns:
            Cleaned content
        """
        import re

        if not content:
            return content

        # First: strip any leaked tool_call markup
        content = cls._strip_tool_call_markup(content)

        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False
        
        def same_title(text: str) -> bool:
            norm = lambda t: re.sub(r'[^a-z0-9]', '', t.lower())
            return norm(text) == norm(section_title)

        for i, line in enumerate(lines):
            stripped = line.strip()

            # The prompt asks for bold text instead of headings, so the LLM
            # sometimes repeats the section title as a bold line (Run 8, section 4).
            bold_match = re.match(r'^\*\*(.+?)\*\*:?$', stripped)
            if i < 5 and bold_match and same_title(bold_match.group(1)):
                skip_next_empty = True
                continue

            # Check if this is a Markdown heading line
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                title_text = heading_match.group(2).strip()
                
                # Check if this heading duplicates the section title (skip duplicates within first 5 lines)
                if i < 5:
                    if same_title(title_text):
                        skip_next_empty = True
                        continue
                
                # Convert all heading levels (#, ##, ###, #### etc.) to bold
                # Since section titles are added by the system, content should not have any headings
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # Add blank line
                continue
            
            # If previous line was a skipped heading and current line is empty, skip it too
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue
            
            skip_next_empty = False
            cleaned_lines.append(line)
        
        # Remove leading blank lines
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)
        
        # Remove leading separator lines
        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)
            # Also remove blank lines after separator
            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)
    
    @classmethod
    def update_progress(
        cls,
        report_id: str,
        status: str,
        progress: int,
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """
        Update report generation progress

        Frontend can read progress.json to get real-time progress
        """
        cls._ensure_report_folder(report_id)
        
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Get report generation progress"""
        path = cls._get_progress_path(report_id)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Get list of generated sections

        Returns information about all saved section files
        """
        folder = cls._get_report_folder(report_id)
        
        if not os.path.exists(folder):
            return []
        
        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse section index from filename
                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])

                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "content": content
                })

        return sections
    
    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        Assemble complete report

        Assemble the full report from saved section files and perform heading cleanup
        """
        # Build report header
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"
        
        # Read all section files in order
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]
        
        # Strip any leaked tool_call markup before post-processing
        md_content = cls._strip_tool_call_markup(md_content)

        # Post-processing: clean up heading issues across the entire report
        md_content = cls._post_process_report(md_content, outline)
        
        # Save complete report
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Complete report assembled: {report_id}")
        return md_content
    
    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        Post-process report content

        1. Remove duplicate headings
        2. Keep report main title (#) and section titles (##), remove other heading levels (###, #### etc.)
        3. Clean up excess blank lines and separators

        Args:
            content: Original report content
            outline: Report outline

        Returns:
            Processed content
        """
        import re
        
        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False
        
        # Collect all section titles from the outline
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check if this is a heading line
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Check if this is a duplicate heading (same content within 5 lines)
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    # Skip duplicate heading and following blank lines
                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue
                
                # Heading level processing:
                # - # (level=1) keep only the report main title
                # - ## (level=2) keep section titles
                # - ### and below (level>=3) convert to bold text
                
                if level == 1:
                    if title == outline.title:
                        # Keep report main title
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # Section title incorrectly used #, correct to ##
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # Other level-1 headings converted to bold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # Keep section title
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # Non-section level-2 headings converted to bold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # ### and below level headings converted to bold text
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False
                
                i += 1
                continue
            
            elif stripped == '---' and prev_was_heading:
                # Skip separator lines immediately following a heading
                i += 1
                continue
            
            elif stripped == '' and prev_was_heading:
                # Keep only one blank line after a heading
                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False
            
            else:
                processed_lines.append(line)
                prev_was_heading = False
            
            i += 1
        
        # Clean up consecutive blank lines (keep at most 2)
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    @classmethod
    def save_report(cls, report: Report) -> None:
        """Save report metadata and complete report"""
        cls._ensure_report_folder(report.report_id)
        
        # Save metadata JSON
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Save outline
        if report.outline:
            cls.save_outline(report.report_id, report.outline)
        
        # Save complete Markdown report
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
        
        logger.info(f"Report saved: {report.report_id}")
    
    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Get report"""
        path = cls._get_report_path(report_id)
        
        if not os.path.exists(path):
            # Legacy format compatibility: check files stored directly in reports directory
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Rebuild Report object
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )
        
        # If markdown_content is empty, try reading from full_report.md
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
        
        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )
    
    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """Get report by simulation ID"""
        cls._ensure_reports_dir()
        
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # New format: folder
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # Legacy format compatibility: JSON file
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report
        
        return None
    
    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        """List reports"""
        cls._ensure_reports_dir()
        
        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # New format: folder
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # Legacy format compatibility: JSON file
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
        
        # Sort by creation time descending
        reports.sort(key=lambda r: r.created_at, reverse=True)
        
        return reports[:limit]
    
    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Delete report (entire folder)"""
        import shutil
        
        folder_path = cls._get_report_folder(report_id)
        
        # New format: delete entire folder
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"Report folder deleted: {report_id}")
            return True
        
        # Legacy format compatibility: delete individual files
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")
        
        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True
        
        return deleted
