# IMPORTING LIBRARIES
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
import pandas as pd
import csv
import os
from dotenv import load_dotenv

# GETTING API KEY FROM .env
_ = load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# SETTING UP EMBEDDINGS AND LANGUAGE MODELS
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-base-en-v1.5",
                                                      model_kwargs={"device": "cuda"})
llm=HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    model_kwargs={"temperature":0.2, "max_length":256},
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

paragraph1 = """
Social anxiety and paranoia often co-occur and exacerbate each other. While loneliness and negative
schemas contribute to the development of social anxiety and paranoia separately, their role in
the development of the two symptoms co-occurring is rarely considered longitudinally. This study
examined the moment-to-moment relationship between social anxiety and paranoia, as well as the
effects of loneliness and negative schemas on both experiences individually and coincidingly. A total
of 134 non-clinical young adults completed experience sampling assessments of momentary social
anxiety, paranoia, and loneliness ten times per day for six consecutive days. Participants’ negativeself
and -other schemas were assessed with the Brief Core Schema Scale. Dynamic structural equation
modelling revealed a bidirectional relationship between social anxiety and paranoia across moments.
Loneliness preceded increases in both symptoms in the next moment. Higher negative-self schema
was associated with a stronger link from paranoia to social anxiety; whereas higher negative-other
schema was associated with a stronger link from social anxiety to paranoia. Our findings support
the reciprocal relationship between social anxiety and paranoia. While loneliness contributes to the
development of social anxiety and paranoia, negative self and other schemas appear to modify the
relationships between the two symptoms.
"""

paragraph2 = """
While previous studies mainly recruited samples across a large age span, the current study focused on
young adulthood (i.e. age 18–30), a life stage where people are most vulnerable to loneliness, social anxiety and
paranoia. The aim of the present study was threefold: First, we tested the moment-to-moment dynamics
between social anxiety and paranoia. We hypothesized significant cross-lagged effects from social anxiety to
paranoia and vice versa. Second, we examined the moment-to-moment dynamics between loneliness and the two
symptoms. We hypothesized significant cross-lagged effects from loneliness to both social anxiety and paranoia.
Third, we tested the associations of core schemas with the strength of the cross-lagged effects. We hypothesized
that negative-self schema would increase the strength of the cross-lagged bi-directional effects between social
anxiety and paranoia. We also hypothesized a positive association between negative-other schema and the
strength of the cross-lagged effect from social anxiety to paranoia.
"""

paragraph3 = """
Artificial intelligence (AI), machine learning and computer vision are revolutionizing research — from medicine and biology to Earth and space
sciences. Now, it’s art history’s turn. For decades, conventionally trained art scholars have been slow to take up computational
analysis, dismissing it as too limited and simplistic. But, as I describe in my book Pixels and Paintings, out this month, algorithms are
advancing fast, and dozens of studies are now proving the power of AI to shed new light on fine-art paintings and drawings.
For example, by analysing brush strokes, colour and style, AI-driven tools are revealing how artists’ understanding of the science of
optics has helped them to convey light and perspective. Programs are recovering the appearance of lost or hidden artworks and
even computing the ‘meanings’ of some paintings, by identifying symbols, for example. It’s challenging. Artworks are complicated
compositionally and materially and are replete with human meaning — nuances that algorithms find hard to fathom.
Most art historians still rely on their individual expertise when judging artists’ techniques by eye, backed up with laboratory, library and
leg work to pin down dates, materials and provenance. Computer scientists, meanwhile, find it easier to analyse 2D photographs or digital
images than layers of oil pigments styled with a brush or palette knife. Yet, collaborations are springing up between computer scientists and art scholars.
Early successes of such ‘computer-assisted connoisseurship’ fall into three categories:
automating conventional ‘by eye’ analyses; processing subtleties in images beyond what is possible through normal human perception;
and introducing new approaches and classes of question to art scholarship. Such methods — especially when enhanced
by digital processing of large quantities of images and text about art — are beginning to empower art scholars, just as microscopes
and telescopes have done for biologists and astronomers.
"""

paragraph4 = """
He knew that Daisy was extraordinary,
but he didn’t realize just how extraordinary a “nice” girl could
be. She vanished into her rich house, into her rich, full life,
leaving Gatsby — nothing. He felt married to her, that was all.
When they met again, two days later, it was Gatsby who was
breathless, who was, somehow, betrayed. Her porch was bright
with the bought luxury of star-shine; the wicker of the settee
squeaked fashionably as she turned toward him and he kissed
her curious and lovely mouth. She had caught a cold, and it
made her voice huskier and more charming than ever, and
Gatsby was overwhelmingly aware of the youth and mystery
that wealth imprisons and preserves, of the freshness of many
clothes, and of Daisy, gleaming like silver, safe and proud
above the hot struggles of the poor.
“I can’t describe to you how surprised I was to find out I
loved her, old sport. I even hoped for a while that she’d throw
me over, but she didn’t, because she was in love with me too.
She thought I knew a lot because I knew different things from
her… . Well, there I was, ‘way off my ambitions, getting deeper
in love every minute, and all of a sudden I didn’t care. What
was the use of doing great things if I could have a better time
telling her what I was going to do?” On the last afternoon before
he went abroad, he sat with Daisy in his arms for a long,
silent time. It was a cold fall day, with fire in the room and her
cheeks flushed. Now and then she moved and he changed his
arm a little, and once he kissed her dark shining hair. The afternoon
had made them tranquil for a while, as if to give them
a deep memory for the long parting the next day promised.
They had never been closer in their month of love, nor communicated
more profoundly one with another, than when she
brushed silent lips against his coat’s shoulder or when he
touched the end of her fingers, gently, as though she were
asleep.
"""

list_of_queries = [
        f"""
        Summarize the paragraph below, delimited by triple backticks, in at most 50 words.
        
        Paragraph: ```{paragraph1}```
        """
    ,
        f"""
        What is the main idea of this paragraph? Describe it in at most 50 words.
        
        Paragraph: ```{paragraph1}```
        """
    ,
        f"""
        Extract all the hypothesis mentioned in the paragraph below
        
        Desired format: 
        Hypothesis1: -||-
        Hypothesis2: -||-
        Hypothesis3: -||-
        Hypothesis4: -||-
        
        Paragraph: ```{paragraph2}```
        """
    ,
        f"""
        List out all the hypothesis mentioned in the paragraph below
        
        Paragraph: ```{paragraph2}```
        """
    ,
        f"""
        What are the hypothesis mentioned in the paragraph below?
        
        Paragraph: ```{paragraph2}```
        """
    ,
        "What is the final sample size consisted of ESM data in the study of examined the moment-to-moment relationship between social anxiety and paranoia?"
    ,
        "In the study of examined the moment-to-moment relationship between social anxiety and paranoia, How many participants are included in the final sample size?"
    ,
        "What is the final sample size consisted of ESM data in the study of examined the moment-to-moment relationship between social anxiety and paranoia?"
    ,
        "What does ESM stand for in the scientific report of The role of loneliness and negative schemas? and How does it work?"
    ,
        "Explain what does ESM stand for and how does it measure paranoia and loneliness according to the scientific report of The role of loneliness and negative schemas"
    ,
        "Based on the FLIGHT OPERATIONS MANUAL (FOM), under what condition, a go around or missed approach has to be executed?"
    ,
        "According to the FLIGHT OPERATIONS MANUAL (FOM), list out 3 conditions that a go around or missed approach has to be executed"
    ,
        "According to the FLIGHT OPERATIONS MANUAL (FOM), a go around should be executed under what situations?"
    ,
        "Apart from Excessive bouncing or pilot-induced oscillations, and Excessive ballooning during round out or flare, what other situation that a pilot must execute a go around approach?"
    ,
        "According to the FLIGHT OPERATIONS MANUAL (FOM), what is the reason of do not attempt to fly under a thunderstorm even if we can see through to the other side?"
    ,
        "When we still can see through to the other side under a thunderstorm, should we attempt to fly? State the reason of why should or why should not?"
    ,
        f"""
        When Taxi out, when the pilot need to test brakes and when do not test brakes?
        Desired format: 
        DO: -||-
        DO NOT: -||-
        """
    ,
        "Does the pilot need to test brakes while transitioning on an active taxiway?"
    ,
        "Under what condition, a pilot is prohibited to test brakes?"
    ,
        f"""
        Your task is to Summarize the text below, in at most 50 words, and focusing on any aspect that 
        are relevant to the benefits of using AI tools in Art study
        
        text: ```{paragraph3}```
        """
    ,
        f"""
        Extract keywords from the below text
        
        text: ```{paragraph3}```
        """
    ,
        f"""
        Extract keywords from the corresponding texts below
        
        Text 1: Computer methods have also recovered missing attributes or portions of incomplete artworks, such as the probable style and colours of ghost paintings
        Keywords 1: recovered, missing attributes, portions of incomplete artworks
        
        Text 2: AI tools can reveal trends in the compositions of landscapes, colour schemes, brush strokes, perspective and more across major art movements.
        Keywords 2: AI tools, trends, compositions of landscapes, colour shemes, brush strokes, perspective
        
        Text 3: {paragraph3}
        Keywords 3:
        """
    ,
        "Why By-eye art analysis can vary depending on how different scholars perceive an artwork? Explain it with example"
    ,
        "Explain how the perception of different sholars on artwork would affect the result of the By-eye art analysis?"
    ,
        "Which regions had more than 3500 millions Total revenues net of interest expense in 2021?"
    ,
        "According to the PARENT COMPANY – CONDENSED STATEMENTS OF INCOME, calculate the net profit margin in 2021 and show me the calculation"
    ,
        "According to the PARENT COMPANY – CONDENSED STATEMENTS OF INCOME, compare the net income between 2020 and 2021"
    ,
        "What is Blue Box Values of American Express"
    ,
        "Summarize and list out all factors to consider when interpreting precipitation forecasts"
    ,
        "Give out 3 factors that need to be consider when interpreting precipitation forecasts"
    ,
        "State the lead time of Seasonal, and Weather(short range) forcast type"
    ,
        "Extract 2 points of each from the Good for understanding, and Not good for understanding under the Seasonal Forecast type from the table of COMPARING PRECIPITATION FORECASTS"
    ,
        "Who was driving the car that hit Myrtle?"
    ,
        "How does Nick Carraway first meet Jay Gatsby?"
    ,
        "What is the relationship between Daisy and Nick?"
    ,
        "Why does Gatsby throw his weekly parties?"
    ,
        "Describe Myrtle’s personality and values in at most 50 words by using details from Chapter 2"
    ,
        f"""
        Use 1 word to describe the relationship between the following characters
        
        Nick and Jordan:
        Nick and Daisy: 
        Daisy and Gatsyby:
        Tom and Myrtle:
        Myrtle and George:
        """
    ,
        f"""
        Identify a list of emotions that Gatsyby is expressing towwards Daisy from the following scene.
        Format the answer as a list of lower-case words separated by commas.
        
        scene: {paragraph4}
        """
    ,
        "What does Anthony Patch tell Gatsyby in the party?"
    ,
        "What did Jordan Baker do after Myrtle was hit by a car?"
    ,
        "Describe the first impression of Jordan Baker from George Wilson aspect"

]

persist_directory = 'db_QA_testing_2000_400'
embedding = HuggingFaceEmbeddings()
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

retriever = vectordb.as_retriever()
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# MAKE A CHAIN TO ANSWER QUERIES
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)

def process_llm_response(llm_response):
    print(llm_response['result'])
    print(llm_response['source_documents'][0].metadata)
    return llm_response['result']

query_answers = {}


with open("db_QA_testing_2000_400.csv", "w", newline="") as csvfile:
    fieldnames = ['Query', 'Answer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in list_of_queries:
        try:
            llm_response = qa_chain(i)
        except RuntimeError:
            print("Took too long...Try again later")
        else:
            query_answers[i] = process_llm_response(llm_response)
            writer.writerow({'Query': i, 'Answer': query_answers[i]})
