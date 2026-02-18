from sklearn.ensemble._stacking import StackingClassifier
from numpy import ndarray
from sentence_transformers import SentenceTransformer
import ollama
from joblib import load

class MentalStateDetector:
    def __init__(self,prompt:str) ->None:
        self.prompt:str = prompt
    
    def analyze_state(self) -> str:
        pipeline:dict = load("model/final_pipeline.joblib")
        embedder:SentenceTransformer = pipeline["embedder"]
        ml_model:StackingClassifier = pipeline["model"]
        X_emb:ndarray = embedder.encode(
            [self.prompt],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        pred_result:ndarray = ml_model.predict(X_emb)
        self.predicted_class:str= str(pred_result[0])
        return self.predicted_class
    
    def prompt_template(self,prompt:str) ->str:
        return f"""
        The users current mental state is: {self.predicted_class}.
        The user has said: {prompt}.
        optimize your response based on the users mental state and provide a helpful response.
        """
    
class LLMModel:
    def __init__(self,refined_prompt:str, model:str, user_mental_state:str) ->None:
        self.refined_prompt:str = refined_prompt
        self.model:str = model
        self.user_mental_state:str = user_mental_state

    def generate_response(self) ->list:
        response:ollama.ChatResponse = ollama.chat(model=self.model,
                    messages=[{"role": "user", "content": self.refined_prompt}])
        result:list = [response["message"]["content"], response['total_duration'] * (10**-9)]
        return result

if __name__ == "__main__":
    model:str ="mistral"
    prompt:str = "I am sad and I want to talk to someone. Can you help me?"
    MSD = MentalStateDetector(prompt)
    result = MSD.analyze_state()
    print(f"Predicted mental state: {result}")
    LLM = LLMModel(MSD.prompt_template(prompt), model, result)
    response = LLM.generate_response()
    print(f"Generated output: \n{response[0]}\nResponse time: {response[1]} seconds")
