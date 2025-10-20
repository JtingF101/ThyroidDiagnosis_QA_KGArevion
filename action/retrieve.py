import logging

class Retrieve:
   def __init__(self, llm, args):
       self.llm = llm
       self.args = args
       # TODO: link db / PubMed API
       # Example self.db = FAISS.load_local(...) or self.pubmed_client = Entrez(...)

   def call(self, triplets, query, max_results=5):
       """
       input: triplets, query
       output: related paper
       """
       logging.info(f"Retrieving literature for: {triplets}")
       keywords_list = []
       for t in triplets:
           if isinstance(t, (list, tuple)) and len(t) == 3:
               s, r, o = t
               keywords_list.append(f"{s} {r} {o}")
           elif isinstance(t, str):
               keywords_list.append(t)
           elif isinstance(t, (list, tuple)):
               keywords_list.append(" ".join(t))
       # If triplets is empty, use query
       if keywords_list:
           keywords = " ".join(keywords_list)
       else:
           keywords = query
       # keywords = " ".join([f"{s} {r} {o}" for (s, r, o) in triplets])
       response = self.llm.generate(f"Find related PubMed papers for: {keywords}", new_tokens_num=512)
       return response

#
# class Retrieve:
#     def __init__(self, llm, args):
#         self.llm = llm
#         self.args = args
#
#     def call(self, triplets, query, max_results=5):
#         """
#         input: triplets, query
#         output: related papers (simulated by LLM)
#         """
#         logging.info(f"Retrieving literature for: {triplets}")
#         keywords_list = []
#
#         for t in triplets:
#             if isinstance(t, (list, tuple)) and len(t) == 3:   # (s, r, o)
#                 s, r, o = t
#                 keywords_list.append(f"{s} {r} {o}")
#             elif isinstance(t, str):
#                 keywords_list.append(t)
#             elif isinstance(t, (list, tuple)):  # Non-triple tuples
#                 keywords_list.append(" ".join(t))
#
#         keywords = " ".join(keywords_list) if keywords_list else query
#
#         # Have the LLM return literature in JSON array format
#         prompt = f"""
#         Based on the biomedical keywords: {keywords},
#         generate {max_results} related PubMed-style references in JSON format.
#
#         Each item should include:
#         - title
#         - authors
#         - journal
#         - year
#         - link
#
#         Return only valid JSON list.
#         """
#
#         response = self.llm.generate(prompt, new_tokens_num=512)
#
#         # If the model fails to generate valid JSON, you can directly return the raw results instead.
#         try:
#             import json
#             papers = json.loads(response)
#             return papers
#         except Exception:
#             logging.warning("LLM did not return valid JSON, fallback to raw text")
#             return response
