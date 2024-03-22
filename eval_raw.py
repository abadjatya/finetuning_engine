import instructor
from openai import OpenAI
import openai
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
import uuid
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Anecdote,ConversationTag,PromptManager,FunctionChoices
from django.db import transaction
# from nltk.corpus import stopwords
# import nltk
from django.contrib.postgres.search import SearchVector
from threading import 
import boto3
import pandas as pd

EXTRACTION_EMBEDDING_MODEL_ID = "text-embedding-ada-002"
EXTRACTION_MODEL_ID = "gpt-3.5-turbo"
TAGGING_MODEL_ID = "gpt-3.5-turbo"

client_embedding = None
client_instructor = None

# hinglish_stopwords = ['a', 'aadi', 'aaj', 'aap', 'aapne', 'aata', 'aati', 'aaya', 'aaye', 'ab', 'abbe', 'abbey', 'abe', 'abhi', 'able', 'about', 'above', 'accha', 'according', 'accordingly', 'acha', 'achcha', 'across', 'actually', 'after', 'afterwards', 'again', 'against', 'agar', 'ain', 'aint', "ain't", 'aisa', 'aise', 'aisi', 'alag', 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'andar', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'ap', 'apan', 'apart', 'apna', 'apnaa', 'apne', 'apni', 'appear', 'are', 'aren', 'arent', "aren't", 'around', 'arre', 'as', 'aside', 'ask', 'asking', 'at', 'aur', 'avum', 'aya', 'aye', 'baad', 'baar', 'bad', 'bahut', 'bana', 'banae', 'banai', 'banao', 'banaya', 'banaye', 'banayi', 'banda', 'bande', 'bandi', 'bane', 'bani', 'bas', 'bata', 'batao', 'bc', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'bhai', 'bheetar', 'bhi', 'bhitar', 'bht', 'bilkul', 'bohot', 'bol', 'bola', 'bole', 'boli', 'bolo', 'bolta', 'bolte', 'bolti', 'both', 'brief', 'bro', 'btw', 'but', 'by', 'came', 'can', 'cannot', 'cant', "can't", 'cause', 'causes', 'certain', 'certainly', 'chahiye', 'chaiye', 'chal', 'chalega', 'chhaiye', 'clearly', "c'mon", 'com', 'come', 'comes', 'could', 'couldn', 'couldnt', "couldn't", 'd', 'de', 'dede', 'dega', 'degi', 'dekh', 'dekha', 'dekhe', 'dekhi', 'dekho', 'denge', 'dhang', 'di', 'did', 'didn', 'didnt', "didn't", 'dijiye', 'diya', 'diyaa', 'diye', 'diyo', 'do', 'does', 'doesn', 'doesnt', "doesn't", 'doing', 'done', 'dono', 'dont', "don't", 'doosra', 'doosre', 'down', 'downwards', 'dude', 'dunga', 'dungi', 'during', 'dusra', 'dusre', 'dusri', 'dvaara', 'dvara', 'dwaara', 'dwara', 'each', 'edu', 'eg', 'eight', 'either', 'ek', 'else', 'elsewhere', 'enough', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'far', 'few', 'fifth', 'fir', 'first', 'five', 'followed', 'following', 'follows', 'for', 'forth', 'four', 'from', 'further', 'furthermore', 'gaya', 'gaye', 'gayi', 'get', 'gets', 'getting', 'ghar', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'good', 'got', 'gotten', 'greetings', 'haan', 'had', 'hadd', 'hadn', 'hadnt', "hadn't", 'hai', 'hain', 'hamara', 'hamare', 'hamari', 'hamne', 'han', 'happens', 'har', 'hardly', 'has', 'hasn', 'hasnt', "hasn't", 'have', 'haven', 'havent', "haven't", 'having', 'he', 'hello', 'help', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', "here's", 'hereupon', 'hers', 'herself', "he's", 'hi', 'him', 'himself', 'his', 'hither', 'hm', 'hmm', 'ho', 'hoga', 'hoge', 'hogi', 'hona', 'honaa', 'hone', 'honge', 'hongi', 'honi', 'hopefully', 'hota', 'hotaa', 'hote', 'hoti', 'how', 'howbeit', 'however', 'hoyenge', 'hoyengi', 'hu', 'hua', 'hue', 'huh', 'hui', 'hum', 'humein', 'humne', 'hun', 'huye', 'huyi', 'i', "i'd", 'idk', 'ie', 'if', "i'll", "i'm", 'imo', 'in', 'inasmuch', 'inc', 'inhe', 'inhi', 'inho', 'inka', 'inkaa', 'inke', 'inki', 'inn', 'inner', 'inse', 'insofar', 'into', 'inward', 'is', 'ise', 'isi', 'iska', 'iskaa', 'iske', 'iski', 'isme', 'isn', 'isne', 'isnt', "isn't", 'iss', 'isse', 'issi', 'isski', 'it', "it'd", "it'll", 'itna', 'itne', 'itni', 'itno', 'its', "it's", 'itself', 'ityaadi', 'ityadi', "i've", 'ja', 'jaa', 'jab', 'jabh', 'jaha', 'jahaan', 'jahan', 'jaisa', 'jaise', 'jaisi', 'jata', 'jayega', 'jidhar', 'jin', 'jinhe', 'jinhi', 'jinho', 'jinhone', 'jinka', 'jinke', 'jinki', 'jinn', 'jis', 'jise', 'jiska', 'jiske', 'jiski', 'jisme', 'jiss', 'jisse', 'jitna', 'jitne', 'jitni', 'jo', 'just', 'jyaada', 'jyada', 'k', 'ka', 'kaafi', 'kab', 'kabhi', 'kafi', 'kaha', 'kahaa', 'kahaan', 'kahan', 'kahi', 'kahin', 'kahte', 'kaisa', 'kaise', 'kaisi', 'kal', 'kam', 'kar', 'kara', 'kare', 'karega', 'karegi', 'karen', 'karenge', 'kari', 'karke', 'karna', 'karne', 'karni', 'karo', 'karta', 'karte', 'karti', 'karu', 'karun', 'karunga', 'karungi', 'kaun', 'kaunsa', 'kayi', 'kch', 'ke', 'keep', 'keeps', 'keh', 'kehte', 'kept', 'khud', 'ki', 'kin', 'kine', 'kinhe', 'kinho', 'kinka', 'kinke', 'kinki', 'kinko', 'kinn', 'kino', 'kis', 'kise', 'kisi', 'kiska', 'kiske', 'kiski', 'kisko', 'kisliye', 'kisne', 'kitna', 'kitne', 'kitni', 'kitno', 'kiya', 'kiye', 'know', 'known', 'knows', 'ko', 'koi', 'kon', 'konsa', 'koyi', 'krna', 'krne', 'kuch', 'kuchch', 'kuchh', 'kul', 'kull', 'kya', 'kyaa', 'kyu', 'kyuki', 'kyun', 'kyunki', 'lagta', 'lagte', 'lagti', 'last', 'lately', 'later', 'le', 'least', 'lekar', 'lekin', 'less', 'lest', 'let', "let's", 'li', 'like', 'liked', 'likely', 'little', 'liya', 'liye', 'll', 'lo', 'log', 'logon', 'lol', 'look', 'looking', 'looks', 'ltd', 'lunga', 'm', 'maan', 'maana', 'maane', 'maani', 'maano', 'magar', 'mai', 'main', 'maine', 'mainly', 'mana', 'mane', 'mani', 'mano', 'many', 'mat', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'mein', 'mera', 'mere', 'merely', 'meri', 'might', 'mightn', 'mightnt', "mightn't", 'mil', 'mjhe', 'more', 'moreover', 'most', 'mostly', 'much', 'mujhe', 'must', 'mustn', 'mustnt', "mustn't", 'my', 'myself', 'na', 'naa', 'naah', 'nahi', 'nahin', 'nai', 'name', 'namely', 'nd', 'ne', 'near', 'nearly', 'necessary', 'neeche', 'need', 'needn', 'neednt', "needn't", 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nhi', 'nine', 'no', 'nobody', 'non', 'none', 'noone', 'nope', 'nor', 'normally', 'not', 'nothing', 'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'par', 'pata', 'pe', 'pehla', 'pehle', 'pehli', 'people', 'per', 'perhaps', 'phla', 'phle', 'phli', 'placed', 'please', 'plus', 'poora', 'poori', 'provides', 'pura', 'puri', 'q', 'que', 'quite', 'raha', 'rahaa', 'rahe', 'rahi', 'rakh', 'rakha', 'rakhe', 'rakhen', 'rakhi', 'rakho', 'rather', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'rehte', 'rha', 'rhaa', 'rhe', 'rhi', 'ri', 'right', 's', 'sa', 'saara', 'saare', 'saath', 'sab', 'sabhi', 'sabse', 'sahi', 'said', 'sakta', 'saktaa', 'sakte', 'sakti', 'same', 'sang', 'sara', 'sath', 'saw', 'say', 'saying', 'says', 'se', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', 'shan', 'shant', "shan't", 'she', "she's", 'should', 'shouldn', 'shouldnt', "shouldn't", "should've", 'si', 'since', 'six', 'so', 'soch', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'still', 'sub', 'such', 'sup', 'sure', 't', 'tab', 'tabh', 'tak', 'take', 'taken', 'tarah', 'teen', 'teeno', 'teesra', 'teesre', 'teesri', 'tell', 'tends', 'tera', 'tere', 'teri', 'th', 'tha', 'than', 'thank', 'thanks', 'thanx', 'that', "that'll", 'thats', "that's", 'the', 'theek', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'theres', "there's", 'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 'thi', 'thik', 'thing', 'think', 'thinking', 'third', 'this', 'tho', 'thoda', 'thodi', 'thorough', 'thoroughly', 'those', 'though', 'thought', 'three', 'through', 'throughout', 'thru', 'thus', 'tjhe', 'to', 'together', 'toh', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'true', 'truly', 'try', 'trying', 'tu', 'tujhe', 'tum', 'tumhara', 'tumhare', 'tumhari', 'tune', 'twice', 'two', 'um', 'umm', 'un', 'under', 'unhe', 'unhi', 'unho', 'unhone', 'unka', 'unkaa', 'unke', 'unki', 'unko', 'unless', 'unlikely', 'unn', 'unse', 'until', 'unto', 'up', 'upar', 'upon', 'us', 'use', 'used', 'useful', 'uses', 'usi', 'using', 'uska', 'uske', 'usne', 'uss', 'usse', 'ussi', 'usually', 'vaala', 'vaale', 'vaali', 'vahaan', 'vahan', 'vahi', 'vahin', 'vaisa', 'vaise', 'vaisi', 'vala', 'vale', 'vali', 'various', 've', 'very', 'via', 'viz', 'vo', 'waala', 'waale', 'waali', 'wagaira', 'wagairah', 'wagerah', 'waha', 'wahaan', 'wahan', 'wahi', 'wahin', 'waisa', 'waise', 'waisi', 'wala', 'wale', 'wali', 'want', 'wants', 'was', 'wasn', 'wasnt', "wasn't", 'way', 'we', "we'd", 'well', "we'll", 'went', 'were', "we're", 'weren', 'werent', "weren't", "we've", 'what', 'whatever', "what's", 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', "where's", 'whereupon', 'wherever', 'whether', 'which', 'while', 'who', 'whoever', 'whole', 'whom', "who's", 'whose', 'why', 'will', 'willing', 'with', 'within', 'without', 'wo', 'woh', 'wohi', 'won', 'wont', "won't", 'would', 'wouldn', 'wouldnt', "wouldn't", 'y', 'ya', 'yadi', 'yah', 'yaha', 'yahaan', 'yahan', 'yahi', 'yahin', 'ye', 'yeah', 'yeh', 'yehi', 'yes', 'yet', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've", 'yup']

# try:
#     stop_words = set(stopwords.words('english'))
# except LookupError as e:
#     nltk.download('stopwords')
#     nltk.download('punkt')
#     stop_words = set(stopwords.words('english'))

# stop_words.update(hinglish_stopwords)

# DATA_TYPE = Literal["subjective", "objective"]
# MOOD_TYPE = Literal[
#     "anger",
#     "fear",
#     "gloomy",
#     "happiness",
#     "romantic",
#     "cheerful",
#     "sadness",
#     "calm",
#     "disgust",
#     "ecstatic",
#     "excited",
#     "idyllic",
#     "lonely",
#     "stressed",
#     "surprise",
#     "annoyed",
#     "bitter",
#     "disappointed",
#     "enjoyment",
#     "mad",
#     "pride",
# ]


# used by gpt
# class Anecdote_GPT(BaseModel):
#     details: str = Field(
#         description="The main content or narrative of the anecdote. It contains the facts or experience of the user."
#     )
#     label: DATA_TYPE = Field(
#         description="The nature of the anecdote, classifying it as either subjective or objective. Subjective anecdotes may involve personal opinions or feelings or experiences, while objective anecdotes typically present factual information."
#     )
#     keywords: List[str] = Field(
#         description="A list of keywords associated with the anecdote. Keywords are terms or phrases that highlight the main themes or topics discussed in the anecdote.This should be sorted according to relevancy."
#     )
#     mood: MOOD_TYPE = Field(
#         description="This captures the mood associated with the anecdote."
#     )
#     # memory_tag: TAG_TYPE = Field(
#     #     description="The tag classification for the anecdote."
#     # )
#     location: Optional[str] = Field(
#         description="The location where the anecdote took place. It provides contextual information about the setting or environment relevant to the narrative."
#     )
#     timestamp: Optional[str] = Field(
#         description="The timestamp associated with the anecdote, indicating the specific date and time when the event or story occurred. If not provided, the timestamp remains unspecified."
#     )
#     day: Optional[str] = Field(
#         description="The day associated with the anecdote.If not provided, the day remains unspecified."
#     )

# class Anecdote_GPT(BaseModel):
#     details:str = Field(description="Explain what the fact is about")
#     location:Optional[str] = Field(description="where did this fact happen.")
#     time:Optional[str] = Field(description="when did this fact happen.")



class UserInformation(BaseModel):
    full_name: Optional[str] = Field(None, description="Full name of the user, including first and last names.")
    age_in_years: Optional[int] = Field(None, ge=0, description="User's age in full years, must be non-negative.")
    current_profession: Optional[str] = Field(None, description="User's current job title or profession.")
    highest_education: Optional[str] = Field(None, description="Highest educational qualification attained by the user.")
    living_location: Optional[str] = Field(None, description="Current living location of the user, including city and country.")
    birthplace: Optional[str] = Field(None, description="Place of birth of the user, typically including city and country.")
    relationship_status: Optional[str] = Field(None, description="Current relationship status of the user, such as single, married, etc.")
    leisure_activities: Optional[List[str]] = Field(None, description="List of leisure activities or hobbies the user regularly engages in.")
    food_preferences: Optional[List[str]] = Field(None, description="List of dietary preferences or restrictions of the user.")
    goals: Optional[List[str]] = Field(None, description="List of personal goals or aspirations the user aims to achieve.")
    preferred_movies: Optional[List[str]] = Field(None, description="List of movies that the user particularly enjoys.")
    favorite_songs: Optional[List[str]] = Field(None, description="List of songs the user counts among their favorites.")
    spoken_languages: Optional[List[str]] = Field(None, description="Languages the user is able to speak.")
    wishlist_destinations: Optional[List[str]] = Field(None, description="Places the user wishes to visit.")
    visited_places: Optional[List[str]] = Field(None, description="Places the user has already visited.")
    date_of_birth: Optional[str] = Field(None, description="User's date of birth, used for age calculation and birthday recognition.")
    style_preferences: Optional[List[str]] = Field(None, description="Preferred styles or brands in fashion by the user.")
    liked_movie_genres: Optional[List[str]] = Field(None, description="Movie genres the user likes.")
    admired_artists: Optional[List[str]] = Field(None, description="Artists across all disciplines (e.g., music, visual arts, literature) that the user admires.")


# class Anecdotes(BaseModel):
#     facts: List[Anecdote_GPT] = Field(
#         description="List of facts about the user that are present in the conversation"
#     )

class Tags(BaseModel):
    tags:List[str] = Field(
        description="A list of maximum five one word labels."
    )


def data_extraction(messages: List[dict], data_model: BaseModel,llm_model:str):
    model = client_instructor.chat.completions.create(
        model=llm_model, response_model=data_model, messages=messages
    )
    return model


def preprocess_messages(
    data: List[dict], incoming_kw: str = "incoming", outgoing_kw: str = "outgoing"
):
    """
    takes input list of messages from the native backend format (json)
    Formats the message from backend to chatml format.
    Extracts the user id and assistant id for accessing the db collection.
    Extracts the list of message id.
    assumption - user id and assistant id remains constant in the list.
    """

    USER_ID = None
    ASSISTANT_ID = None
    formatted_messages = []
    message_ids = []

    for i, entity in enumerate(data):
        if i == 0:
            USER_ID = entity["user_id"]
            ASSISTANT_ID = entity["assistant_id"]

        curr_message = {}
        if entity["direction"] == incoming_kw:
            curr_message = {"role": "user", "content": entity["message"]}
        elif entity["direction"] == outgoing_kw:
            curr_message = {"role": "assistant", "content": entity["message"]}

        formatted_messages.append(curr_message)
        message_ids.append(entity["id"])

    return formatted_messages, message_ids, USER_ID, ASSISTANT_ID


def get_system_prompt(function_name:str):
    obj = PromptManager.objects.filter(function=function_name).latest('created_at')
    instruction = obj.instruction
    examples = obj.examples
    prompt_id = obj.id
    final_prompt = instruction
    if len(examples) == 0:
        system_message = {"role": "system", "content": final_prompt}
        return system_message,prompt_id
    final_prompt += "\nExamples:\n"
    for i,example in enumerate(examples):
        formatted_example = f"{i+1}. {example}\n"
        final_prompt += formatted_example
    system_message = {"role": "system", "content": final_prompt}
    return system_message,prompt_id



# api endpoint to accept golden dataset
# fetches the dimension prompt and instructions 
# run the testcases from the golden dataset
# return a sheet with score and reason.
@api_view(["POST"])
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.dataset import EvaluationDataset


s3 = boto3.client('s3', aws_access_key_id=settings.AWS_ACCESS_KEY_ID, aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)
obj = s3.get_object(Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key='data.csv')
data = obj['Body'].read().decode('utf-8')
df = pd.read_csv(pd.compat.StringIO(data))

curr_dataset = EvaluationDataset()
inputs = df["inputs"]
expected_outputs = df["expected_outputs"]

# run the inference engine for the given model id and create the actual_outputs
# use the vllm inference engine to get actual outputs (batching)
# start the inference engine on runpod 
# access models from huggingface
# run the batched inference on all the inputs




actual_outputs = None 

for _input,actual_output,expected_output in zip(inputs,actual_outputs,expected_outputs):
    curr_test_case = LLMTestCase(input=_input,actual_output=actual_output,expected_output=expected_output)
    curr_dataset.add_test_case(curr_test_case)

# @TODO 
for curr_metric in metric_list:
    #check if the config passes the sanity check.
    config = PromptManager.objects.filter(function=curr_metric).latest('created_at')
    # redefine the prompt manager to be more agnostic
    #met

    metric_name = config["dimension"]
    thought_chain = config["thought_chain"]
    params_list = config["params_list"] # input , output(actual and ideal)
    threshold = config["threshold"]

    curr_metric = GEval(
        name = metric_name,
        evaluation_steps = thought_chain,
        evaluation_params = params_list,
        threshold = threshold
    )

    curr_dataset.evaluate([curr_metric])

    #run this metric on the dataset efficiently

    #give a csv in return with existing columns 
    #and score and reason as added columns.



metric_config = 

coherence_metric = GEval(
    name="Coherence",
    criteria="Coherence - determine if the actual output is coherent with the input.",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    evaluation_steps=["Check whether the sentences in 'actual output' aligns with that in 'input'"],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)





@api_view(["POST"])
def handler(request):
    global client_embedding, client_instructor

    if client_embedding is None or client_instructor is None:
        client_embedding = OpenAI()
        client_instructor = instructor.patch(OpenAI())

    messages = request.data
    try:
        messages_formatted, MESSAGE_ids, USER_id, ASSISTANT_id = preprocess_messages(
            messages
        )
        extractor_messages = messages_formatted
        print("Messages formatting done")
    except Exception as e:
        print(e)
        return Response(e, status=status.HTTP_400_BAD_REQUEST)

    #removing assistant messages


    # after this portion running inside thread from now.
    def extractor_background():
        system_message,extractor_prompt_id = get_system_prompt(FunctionChoices.EXTRACTOR)
        final_extractor_messages = [system_message,{"role":"user","content":str(extractor_messages)}]
        print("Extraction using GPT started.")
        try:
            res = data_extraction(final_extractor_messages,UserInformation,EXTRACTION_MODEL_ID)
        except Exception as e:
            print(f"Unexpected Error for extraction Occured: {e}")
        print("Extraction using GPT done.")
        # adding ids to each anecdote object and creating embedding for the same using openai model.
        db_data = []
        extracted_data = res.dict()
        for field_name, value in extracted_data.items():

            if value is None:
                continue

            if isinstance(value, list):
                if len(value) == 1:
                    embedding_text = f"User's {field_name} is {str(value[0])}"
                else:
                    string_list = [str(item) for item in value]
                    embedding_text = f"User's {field_name} are {', '.join(string_list)}"
            else:
                embedding_text = f"User's {field_name} is {str(value)}"

            fault_memories = ["User's full_name is Priya", "User's full_name is priya", "User's full_name is Zoya", "User's full_name is zoya", "User's full_name is Shweta", "User's full_name is shweta"]
            if embedding_text in fault_memories:
                continue

            fault_substrings = ["not provided", "not specified"]
            for substring in fault_substrings:
                if substring in embedding_text.lower():
                    continue

            try:
                curr_embedding_obj = client_embedding.embeddings.create(
                    input=[embedding_text], model=EXTRACTION_EMBEDDING_MODEL_ID
                )
            except Exception as e:
                print(f"Unexpected Error for embedding Occured: {e}")

            curr_embedding = curr_embedding_obj.data[0].embedding
            curr_data = {
                "user_id": USER_id,
                "assistant_id": ASSISTANT_id,
                "details": embedding_text,
                "message_ids": MESSAGE_ids,
                "embedding": curr_embedding,
                "prompt_id": extractor_prompt_id,
                "extractor_llm_model":EXTRACTION_MODEL_ID,
                "extractor_embedding_model":EXTRACTION_EMBEDDING_MODEL_ID
            }
            curr_obj = Anecdote(**curr_data)
            db_data.append(curr_obj)

        try:
            with transaction.atomic():
                pushed_data = Anecdote.objects.bulk_create(db_data)
                print("DB entry of extracted data successful")
        except Exception as e:
            print(f"Error during creating db entries for extraction : {e}")

    def tag_background():
        system_message,conversation_tag_prompt_id = get_system_prompt(FunctionChoices.CONVERSATION_TAGS)
        final_tagging_messages = [system_message,{"role":"user","content":str(messages_formatted)}]
        print("Conversation tagging using GPT started.")
        try:
            res = data_extraction(final_tagging_messages,Tags,TAGGING_MODEL_ID)
        except Exception as e:
            print(f"Unexpected Error for tagging occured: {e}")
        print("Conversation tagging using GPT done.")

        curr_tag_data = {
            "user_id": USER_id,
            "assistant_id": ASSISTANT_id,
            "tags": res.tags,
            "message_ids": MESSAGE_ids,
            "prompt_id": conversation_tag_prompt_id,
            "tagging_llm_model": TAGGING_MODEL_ID
        }

        try:
            obj = ConversationTag(**curr_tag_data)
            obj.save()
            print("DB entry of tags successful.")
        except Exception as e:
            print(f"Unexpected Error Occured during conversation tagging db write: {e}")


    extractor_thread = Thread(target=extractor_background)
    tag_thread = Thread(target=tag_background)

    extractor_thread.start()
    tag_thread.start()
    print("Started both threads.")

    return Response("Extractor has started",status=status.HTTP_200_OK)


# @api_view(["POST"])
# def populate_stopwords(request):
#     words = request.data
#     db_data = []
#     word_list = ast.literal_eval(words["words_list"])
#     for curr_word in word_list:
#         curr_data = {
#             "word":curr_word
#         }
#         curr_obj = Stopwords(**curr_data)
#         db_data.append(curr_obj)

#     try:
#         with transaction.atomic():
#             Stopwords.objects.bulk_create(db_data)
#         return Response("DB ENTRIES SUCCESSFUL", status=status.HTTP_200_OK)
#     except Exception as e:
#         return Response(str(e), status=status.HTTP_400_BAD_REQUEST)


# @api_view(["POST"])
# def vector_search_handler(request):
#     global client_embedding

#     if client_embedding is None:
#         client_embedding = OpenAI()

#     recieved_data = request.data
#     snippet = recieved_data["message"]
#     user_id = recieved_data["user_id"]
#     assistant_id = recieved_data["assistant_id"]


#     curr_embedding_obj = client_embedding.embeddings.create(
#         input=[snippet], model=EMBEDDING_MODEL_ID
#     )
#     curr_embedding = curr_embedding_obj.data[0].embedding

#     try:
#         result = Anecdote.objects.filter(user_id=user_id, assistant_id=assistant_id).order_by(L2Distance('embedding', curr_embedding))[:1]
#         return Response(result[0].details, status=status.HTTP_200_OK)
#     except Exception as e:
#         return Response(str(e), status=status.HTTP_400_BAD_REQUEST)
