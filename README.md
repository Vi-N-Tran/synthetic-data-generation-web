# Composite Technical Assessment: Browser Trajectory Dataset Generation

## Overview
Design and implement a synthetic dataset generation pipeline for training browser automation models. This assessment evaluates your ability to translate real-world user behavior into structured data suitable for model training.

**Time Expectation:** 3 hours  

## AI Guidelines
You may use LLM's during this assignment to implement your code. If you choose to do so, you are expected to defend design decisions, sytnax choices, and implementation details as your own. 

**All work should be done within a single screen** (which is being shared and recorded over Zoom). All other screens must be disconnected from your device prior to starting the assessment. Multiple monitors are NOT permitted.


## Context
Composite builds intelligent browser automation that handles repetitive tasks on users' behalf. To predict and recommend user actions effectively, we rely on high-quality trajectory data that captures realistic browsing patterns and user intent.

## Your Task
Build a data generation pipeline that creates synthetic browser interaction trajectories. Your solution should demonstrate both technical implementation skills and ML system design thinking. 

### Language Requirements 

You may use either **Python** or **Typescript / Javascript** as your language of choice for this assessment. Your language choice will not impact your scores. 

### Core Requirements

1. **Data Schema Design** 
   - Define what constitutes a browser "action" (clicks, typing, navigation, etc.)
   - Specify features that capture user intent and context
   - Include temporal relationships between actions
   - Consider what metadata would be valuable for downstream ML tasks

2. **Data Structure & Storage**
   - Define or implement a storage format for trajectory sequences
   - Balance between human readability and computational efficiency

3. **Synthetic Data Generation**
   - Implement realistic trajectory generation logic
   - Include common user workflows (e.g., form filling, shopping, research)
   - Add controlled variability to prevent overfitting
   - Generate at least 1000 trajectories with varying action lengths (3 - 10 steps)

4. **Data Quality & Validation**
   - Implement checks for data consistency and validity
   - Include statistics on generated dataset (action distributions, trajectory lengths)
   - Demonstrate how to detect and handle edge cases

### Deliverables

1. **Code Implementation**
   - Python or Typescript / Javascript implementation (PyTorch/TensorFlow experience is a plus)
   - Clear project structure with modular components
   - Requirements.txt / environment.yml or package.json

2. **Generated Dataset**
   - Sample of 1000 trajectories in your chosen format
   - Summary statistics of the full generated dataset

3. **Documentation**
   - README explaining your design decisions and trade-offs
   - Data schema documentation with field descriptions
   - Brief analysis of how this data would be used for model training


## Evaluation Criteria

We'll assess your submission based on:

- **System Design (30%)**: Data schema appropriateness, scalability considerations, ML-readiness
- **Implementation Quality (30%)**: Code organization, efficiency, best practices
- **Problem Solving (20%)**: Handling edge cases, data quality measures, creative solutions
- **Documentation (20%)**: Clarity of technical decisions, reproducibility, thoughtful analysis


## Resources 

You will be given a OpenAI API key with $20.00 in credit that may be used in the course of data generation. 

## Technical Considerations

Consider addressing these aspects in your solution:
- How would you handle different types of websites (e-commerce, social media, productivity tools)?
- How do you handle different types of elements (canvases, iframes, single page applications)
- What features would help a model distinguish between intentional actions and exploration?
- How do you balance synthetic data realism with generation efficiency?
- What would change if this pipeline needed to process real user data instead?

## Submission Instructions

1. Implement your solution in either the @python or @typescript directory
2. Include all code, documentation, and sample data
4. Send submission to careers@composite.com with any additional notes or clarifications

## Questions?

This assessment is intentionally open-ended to see how you approach ambiguous problems. However, if you have clarifying questions about requirements or constraints during the assessment, feel free to call Charlie (+16178510006) or Yang (+16506805064)

---

**Note on Time Management:** We expect this assessment to take approximately 3 hours to complete. Quality is more important than speed, so take the time you need to deliver a program you're proud of. If you do not finish within the 4 hours, that's completely fine - your quality of thinking and clearly communicating to us what you've tried is more important.

Good luck, and we look forward to seeing your work!



## Sources and Further Reading

- [Web Arena](https://webarena.dev/)
- [OS-Gensis](https://github.com/OS-Copilot/OS-Genesis)
- [Online-Mine2Web](https://github.com/OSU-NLP-Group/Online-Mind2Web)
- [Playwright Actions](https://playwright.dev/docs/input)