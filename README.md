# Retrieval Augmented Generation with Vectors

This project was started to first learn how Retrieval Augmented Generation can work with Vectors to get information from unstructured data (like documents or pdfs). After researching and figuring out how we can actually use Vectors, we wanted to apply them to Large Language Models. 

Our idea was that a lot of the stable LLMs out in the open are made from many big datasets from the past, which means there will usually be a cut-off point where some new data will not be understood by the LLM. This is where we can apply such vectors and effectively add them to the huge repository of data so that the LLM can seamlessly generate new text with the additional knowledge.

The method we used was that we'd pick a LLM, and then we looked for some unstructured data (mainly pdfs with a more focused knowledge of a certain subject) and also experimented with chunk sizes and chunk overlaps. Additionally we tried to ask the same question but in a different way to see if the chatbot could answer based on the information we fed (instead of some older information, mainly applies for things like Acronyms).

You could find some examples from these csv/xlsx files:
https://drive.google.com/drive/u/0/folders/1R3A4fHQQj_th1a6gMK8zf9-dAAqJqtKV

To run the chatbot:
1. pip install -r requirements.txt
2. Create a .env file for API Token
3. Run main.py

You can also change which database to use.

If you want to add in new unstructured data, you can move your selected PDF into the PDF_doc folder first.

