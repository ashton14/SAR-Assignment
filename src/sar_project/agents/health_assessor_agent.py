from sar_project.agents.base_agent import SARBaseAgent
import os
import google.generativeai as genai

from dotenv import load_dotenv
import sys

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Add the `src` directory to sys.path to allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


class HealthAssessorAgent(SARBaseAgent):
    def __init__(self, name="health_assessor"):
        super().__init__(
            name=name,
            role="Health Assessor",
            system_message="""You are a health assessor for SAR operations. Your role is to:
            1. Assess likely health status of missing person
            2. Determine necessary medications
            3. Determine urgency of necessary treatment
            4. Assess mobility of missing person
            5. Extract medical conditions from medical history"""
        )


#Only works with python version > 3.10
    def query_gemini(self, prompt, model="gemini-pro", max_tokens=200):
        """Query Google Gemini API and return response."""
        try:
            response = genai.GenerativeModel(model).generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {e}"

    def process_request(self, message):
        """Process health-related requests"""
        try:
            if "assess_health_risks" in message:
                return self.assess_health_risks(message["conditions"])
            elif "get_medications" in message:
                return self.get_medications(message["injury"])
            elif "assess_urgency" in message:
                return self.assess_urgency(message["injury"])
            elif "evaluate_mobility" in message:
                return self.evaluate_mobility(message["conditions"])
            elif "extract_medical_conditions" in message:
                return self.extract_medical_conditions(message["medical_history_form"])
            else:
                return {"error": "Unknown request type"}
        except Exception as e:
            return {"error": str(e)}


    def assess_health_risks(self, conditions):
        """Assess likely current health risks of missing person
           Input: List of medical/health conditions
           Output: Potential health risks or problems that can be caused
                   by the listed conditions.
                   
            ex. [High blood pressure, athsma] ->
                    Lightheadedness, shortness of breath, fainting
        """

        result = {}

        for condition in conditions:

            prompt = f"Give me a concise list of potential risks from having {condition}"
            response = self.query_gemini(prompt)
            result[condition] = response

        return result
    

    def get_medications(self, injury):
        """Get medications necessary for treatment
           Input: An injury
           Output: Medications that can be used to treat the listed injury.

           ex. open cut wound ->  Bacitracin, Tylenol, Aquaphor
        """

        prompt = f"Give me a list of medications required to treat {injury} \
                    in a comma-separated list."

        response = self.query_gemini(prompt)  
        return {
            "medications": response,
        }


    def assess_urgency(self, injury):
        """Assess urgency of treatment
           Input: An injury
           Output: The urgency of treating the injury as one of the following:
                   1) extremely urgent
                   2) fairly urgent
                   3) minor urgency
                   4)not urgent
                   
            ex. Gunshot wound -> Extremely urgent
           """

        prompt = f"Tell me the urgency of treatment for a {injury} \
            as either extremely urgent, fairly urgent, minor urgency, \
                or not urgent in two words."

        response = self.query_gemini(prompt)  
        return {
            "urgency": response,
        }
    
    def evaluate_mobility(self, conditions):
        """Assess mobility of the missing person based on medical conditions
           Input: A list of medical/health conditions
           Output: The potential mobility of the person given the conditions
                   as one of the following: 
                   1) immobile
                   2) slightly mobile
                   3) fairly mobile
                   4) very mobile

           ex. [sprained ankle, athsma] -> slightly mobile
                   
        """

        result = {}

        conditions = ", ".join(conditions)

        prompt = f"based on these conditions, assess whether this person's mobility. \
            return one of the following: immobile, slightly mobile, fairly mobile, very mobile: \
                {conditions}"
        response = self.query_gemini(prompt)

        return {
            "mobility": response
        }
    

    def extract_medical_conditions(self, medical_history_form):
        """Extract list of medical conditions given the person's medical history form
           Input: A medical history form
           Output: A concise list of medical conditions
        """

        prompt = f"Extract a concise list of medical conditions/problems \
            from the given medical history form: {medical_history_form}"

        response = self.query_gemini(prompt)  
        return {
            "conditions": response,
        }


    def extract_allergies(self, medical_history_form):
        """Extract list of allergies given the person's medical history form
           Input: A medical history form
           Output: A concise list of allergies
        """

        prompt = f"Extract a concise list of allergies \
            from the given medical history form: {medical_history_form}"

        response = self.query_gemini(prompt)  
        return {
            "allergies": response,
        }
    

    def update_status(self, status):
        """Update the agent's status"""
        self.status = status
        return {"status": "updated", "new_status": status}

    def get_status(self):
        """Get the agent's current status"""
        return getattr(self, "status", "unknown")
