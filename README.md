# DeepSearch Code

`deepsearch-code` is designed to answer deep, complex questions about large code repositories using LLMs. Rather than trying to fit the entire codebase into the context window, it uses an agent loop that allows the LLM to search and read files within the repository. It's made of two "agents":
- the "manager" agent - This agent has two primary responsibilities: asking questions and returning the final response to your query.
- the "researcher" agent - This agent is what searches and reads files from the responsitory to answer the manager's questions.
The idea is that the manager operates at a higher level and asks the right questions to get the information for a comprehensive response to the input question. The researcher looks at the actual code, and responds to each question. Different models can be used for the two--the researcher tends to use more tokens than the manager, so I've gotten pretty good results using a smaller, faster model like `google/gemini-2.0-flash-001` for the researcher model. Using a high-end model such as `google/gemini-2.5-pro-preview-03-25` for the manager tends to yield the best results (and in fact, those two are the default models for each role).

## Running

You can run `deepsearch-code` by doing one of the following:
```bash
# pip - use a virtualenv etc.
pip install git+https://github.com/cfeenstra67/deepsearch-code
deepsearch-code
# uv
uv add git+https://github.com/cfeenstra67/deepsearch-code
deepsearch-code <args>
# uvx
uvx --from git+https://github.com/cfeenstra67/deepsearch-code deepsearch-code
```
