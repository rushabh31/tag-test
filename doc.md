# AI-Assisted Investigation System Documentation

## Overview

This documentation describes a comprehensive AI-powered investigation and data analysis system designed to automate complex investigation workflows through intelligent agent coordination and semantic data operations. The system employs a two-phase approach combining human oversight with automated execution to deliver thorough, actionable investigation results.

## System Architecture

### Core Philosophy
The system is built on a multi-agent architecture where specialized AI agents collaborate under the supervision of a central coordinator to execute complex investigation plans. Each agent has distinct capabilities and responsibilities, enabling parallel processing and specialized handling of different data sources and operations.

## Two-Phase Operational Framework

### Phase 1: AI-Assisted Investigation Plan Generation

**Objective**: Generate detailed, step-by-step investigation plans based on user queries and available knowledge resources.

**Process Flow**:
1. **Input Collection**: Users provide investigation queries through a chatbot-like UI interface
2. **Knowledge Integration**: The system leverages its knowledge base including:
   - Business Guidance documentation
   - System & Fields Metadata
   - Existing action/patterns information
3. **Plan Generation**: An LLM generates a comprehensive, actionable investigation plan using curated prompt templates
4. **Human Review**: Users review the generated plan and can provide feedback or request modifications
5. **Plan Approval**: Once satisfied, users approve the plan to proceed to Phase 2

**Key Features**:
- Interactive chatbot interface for intuitive query input
- Integration with comprehensive knowledge bases
- Human-in-the-loop validation ensures plan accuracy and relevance
- Flexible plan modification based on user feedback

### Phase 2: Automated Semantic Execution using AI Agents

**Objective**: Execute the approved investigation plan through coordinated agent actions across relevant data sources.

**Process Flow**:
1. **Plan Parsing**: The SupervisorAgent parses the approved investigation plan
2. **Agent Coordination**: Based on plan requirements, the supervisor coordinates appropriate agents
3. **Parallel Execution**: Multiple agents execute their assigned tasks simultaneously
4. **Data Integration**: Results from various agents are consolidated in the global state
5. **Report Generation**: Final comprehensive report is generated based on all collected data

**Key Features**:
- Automated execution reduces manual effort and human error
- Semantic operators enable natural language-based data operations
- Parallel processing improves efficiency
- Comprehensive result aggregation and reporting

## Agent Architecture

### SupervisorAgent
**Primary Role**: Central coordinator responsible for workflow management, state coordination, and inter-agent communication.

**Key Responsibilities**:
- Interpreting investigation plans and determining required agent actions
- Managing the global state and ensuring data consistency
- Coordinating between worker agents based on current state and user requests
- Orchestrating the overall investigation workflow
- Verifying completion of investigation steps
- Generating final execution reports

**Decision Logic**: The SupervisorAgent uses the current state and user request to determine which specific agent should execute the next required action.

### IOAgent (Input/Output Agent)
**Primary Role**: Handles all data input operations from various sources.

**Available Operations**:
- `get_data_from_excel`: Loads and processes data from Excel files
- `get_data_from_csv`: Loads and processes data from CSV files  
- `get_data_using_api`: Fetches data from external systems using APIs

**Key Responsibilities**:
- Consuming data requests and loading data to the global state
- Handling different file formats and data sources
- Ensuring data integrity during the loading process
- Managing connection protocols for various data sources

### REPLAgent (Read-Eval-Print Loop Agent)
**Primary Role**: Performs data transformation and analysis using Python code execution.

**Available Operations**:
- `Execute_code`: Executes provided Python code in a REPL environment

**Key Responsibilities**:
- Transforming data using REPL (Read-Eval-Print Loop) operations
- Consuming transformation requests and proposing appropriate Python code
- Executing code in a controlled environment
- Storing transformation results in the global state
- Maintaining execution status across multiple function calls

### ReportAgent
**Primary Role**: Generates comprehensive final reports based on investigation results.

**Key Responsibilities**:
- Consuming investigation results from the global state
- Generating structured, comprehensive reports
- Synthesizing findings from multiple data sources
- Presenting results in user-friendly formats
- Ensuring report completeness and accuracy

### SemanticAgent
**Primary Role**: Handles semantic requests and natural language-based data operations.

**Available Semantic Operations**:

#### Data Mapping and Transformation
- `sem_map`: Maps each record using natural language projections
- `sem_extract`: Extracts one or more attributes from each row using natural language queries

#### Filtering and Selection
- `sem_filter`: Keeps records that match natural language predicates
- `sem_search`: Performs semantic search over text columns

#### Aggregation and Summarization
- `sem_agg`: Aggregates across all records (e.g., for summarization purposes)

#### Sorting and Ranking
- `sem_topk`: Orders records by natural language sorting criteria

#### Data Joining
- `sem_join`: Joins two datasets based on natural language predicates
- `sem_sim_join`: Joins two DataFrames based on semantic similarity

**Key Responsibilities**:
- Processing semantic requests using natural language specifications
- Determining appropriate semantic operators for given requests
- Enabling intuitive, natural language-based data manipulation
- Bridging the gap between human intent and data operations

## Example Investigation Workflow

### Scenario
**Event Occurrence**: Developer utilized the FWS application (ID 123456) "impersonation" feature, logged in as a trader, and accidentally cancelled a trade transaction worth $10MM.

### Generated Investigation Plan

1. **Operational Events Review**
   - Review Operational Events data from ABC to find and understand the specific event
   - Identify patterns and anomalies related to the impersonation feature usage

2. **Application Analysis**
   - Review BCD data for applications with "impersonation" login features
   - Identify each application's deployment status (production or development)
   - Map the scope of applications with similar vulnerability

3. **Access Control Assessment**
   - Review EFG entitlements data for each identified application
   - Determine the number of individuals with access to impersonation features
   - Assess the magnitude of potential risk exposure

4. **Deployment Status Verification**
   - Verify each application status listed in ID systems
   - Cross-reference with active server locations in GFT
   - Identify true application deployment status across environments

5. **Governance Review**
   - Review data from ORM Governance systems
   - Ensure compliance with established access control policies
   - Identify any governance gaps or violations

6. **Issue Tracking and Remediation**
   - Review IST data to find associated issues and CAP (Corrective Action Plans)
   - Understand existing remediation efforts
   - Develop comprehensive remediation strategy

### Execution Flow

1. **Initiation**: SupervisorAgent receives the approved investigation plan
2. **Data Collection**: IOAgent fetches data from ABC, BCD, EFG, GFT, ORM, and IST systems
3. **Semantic Analysis**: SemanticAgent processes operational events data to identify patterns related to the impersonation feature
4. **Data Processing**: REPLAgent performs data transformations and analysis
5. **Cross-Reference**: SemanticAgent joins datasets to verify application statuses and access controls
6. **Verification**: Each step is verified by the SupervisorAgent before proceeding
7. **Report Generation**: ReportAgent synthesizes all findings into a comprehensive investigation report

## Technical Implementation Details

### Global State Management
- Centralized state storage ensures data consistency across all agents
- All agents read from and write to the shared global state
- State management enables tracking of investigation progress and intermediate results

### Agent Communication Protocol
- SupervisorAgent acts as the central communication hub
- Agents communicate through the global state rather than direct inter-agent communication
- Structured request/response patterns ensure reliable agent coordination

### Error Handling and Resilience
- Each agent maintains its execution status for reliability
- Failed operations can be retried without affecting the overall investigation
- Partial results are preserved to enable investigation continuation

### Natural Language Processing
- Semantic operations leverage advanced NLP capabilities
- Natural language specifications are translated into appropriate data operations
- Context-aware processing ensures accurate interpretation of user intent

## Benefits and Advantages

### Efficiency Gains
- Automated execution significantly reduces manual investigation time
- Parallel agent processing enables faster completion of complex investigations
- Semantic operations eliminate the need for complex query writing

### Accuracy Improvements
- Systematic approach ensures comprehensive coverage of investigation areas
- Automated verification reduces human error
- Consistent methodology across different types of investigations

### Scalability
- Multi-agent architecture supports handling multiple concurrent investigations
- Modular design allows for easy addition of new agent types and capabilities
- Semantic operators provide flexible adaptation to new investigation requirements

### User Experience
- Natural language interface reduces technical barriers
- Human-in-the-loop design maintains user control and oversight
- Comprehensive reporting provides actionable insights and clear next steps

## Future Enhancements

### Potential Expansions
- Additional agent types for specialized data sources
- Enhanced semantic operations for more complex data manipulations
- Integration with additional external systems and APIs
- Advanced visualization capabilities for investigation results

### Continuous Improvement
- Machine learning integration for pattern recognition and anomaly detection
- Automated plan optimization based on historical investigation outcomes
- Enhanced natural language understanding for more intuitive user interactions

This AI-Assisted Investigation System represents a significant advancement in automated investigation capabilities, combining the power of AI agents with human oversight to deliver comprehensive, accurate, and efficient investigation results.
