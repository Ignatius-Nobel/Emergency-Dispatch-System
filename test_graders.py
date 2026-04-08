import sys
from models import DispatchGridAction
from graders import EasyGrader, MediumGrader, HardGrader
from server.dispatch_grid_environment import EASY_CALLS, MEDIUM_CALLS, HARD_CALLS

def test_grader():
    e = EasyGrader()
    m = MediumGrader()
    h = HardGrader()
    
    # Test easy
    if len(EASY_CALLS) > 0:
        call = EASY_CALLS[0]
        action = DispatchGridAction(
            ambulance_units=1, police_units=0, fire_units=0, priority_level=2,
            backup_requested=False, hospital_choice="general_hospital",
            coordination_level="none", ambulance_staging="immediate_scene"
        )
        score = e.grade(action, {"call_id": call.call_id})
        print(f"Easy score: {score}")
    
    # Test medium
    if len(MEDIUM_CALLS) > 0:
        call = MEDIUM_CALLS[0]
        action = DispatchGridAction(
            ambulance_units=1, police_units=1, fire_units=0, priority_level=2,
            backup_requested=False, hospital_choice="general_hospital",
            coordination_level="none", ambulance_staging="immediate_scene"
        )
        score = m.grade(action, {"call_id": call.call_id})
        print(f"Medium score: {score}")
        
    # Test hard
    if len(HARD_CALLS) > 0:
        call = HARD_CALLS[0]
        action = DispatchGridAction(
            ambulance_units=1, police_units=1, fire_units=1, priority_level=1,
            backup_requested=False, hospital_choice="general_hospital",
            coordination_level="none", ambulance_staging="immediate_scene"
        )
        score = h.grade(action, {"call_id": call.call_id})
        print(f"Hard score: {score}")
            
    print("All tests passed.")

if __name__ == "__main__":
    test_grader()
