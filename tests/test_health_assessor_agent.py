import pytest
from sar_project.agents.health_assessor_agent import HealthAssessorAgent

class TestHealthAssessorAgent:
    @pytest.fixture
    def agent(self):
        return HealthAssessorAgent()

    def test_initialization(self, agent):
        assert agent.name == "health_assessor"
        assert agent.role == "Health Assessor"
        assert agent.mission_status == "standby"

    def test_process_request(self, agent):
        message = {
            "assess_health_risks": True,
            "conditions": ["asthma", "pneumonia"]
        }
        response = agent.process_request(message)
        assert "asthma" in response
        assert "pneumonia" in response

    def test_get_medications(self, agent):
        response = agent.get_medications("Cut on foot")
        assert "medications" in response

    def test_assess_urgency(self, agent):
        response = agent.assess_urgency("Gunshot wound")
        assert "urgency" in response

    def test_assess_health_risks(self, agent):
        conditions = ["asthma", "diabetes"]
        response = agent.assess_health_risks(conditions)
        assert "asthma" in response
        assert "diabetes" in response

    def test_evaluate_mobility(self, agent):
        conditions = ["asthma", "diabetes", "influenza"]
        response = agent.evaluate_mobility(conditions)
        assert "mobility" in response

    def test_extract_medical_conditions(self, agent):
        form = "Sarah, a 42-year-old woman, has been struggling with a combination \
            of high blood pressure and frequent migraines. Her blood pressure readings \
            consistently hover around 150/95 mmHg, placing her in the stage 2 \
            hypertension category. These elevated readings often coincide with \
            the onset of her migraines, which she experiences at least twice \
            a week. \
            Sarah's migraines are characterized by throbbing pain on one side of\
            her head, accompanied by nausea, sensitivity to light and sound, and \
            visual disturbances like flashing lights. The migraines typically last \
            for several hours, significantly disrupting her daily activities and \
            causing considerable distress."
        response = agent.extract_medical_conditions(form)
        assert "conditions" in response


    def test_status_update(self, agent):
        response = agent.update_status("active")
        assert response["new_status"] == "active"
        assert agent.get_status() == "active"

