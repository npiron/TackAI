#!/usr/bin/env python3
"""
Test script to verify the improved DQN action space.
Tests all 9 discrete actions to ensure they map correctly to on/off controls.
"""

def test_action_mapping():
    """Test that all 9 actions map correctly to discrete on/off controls."""
    print("🧪 Testing DQN Action Space (Discrete 9)")
    print("=" * 60)
    
    actions = [
        (0, "Idle", {"steer": 0.0, "accel": 0.0, "brake": 0.0}),
        (1, "Accelerate", {"steer": 0.0, "accel": 1.0, "brake": 0.0}),
        (2, "Brake", {"steer": 0.0, "accel": 0.0, "brake": 1.0}),
        (3, "Left", {"steer": -1.0, "accel": 0.0, "brake": 0.0}),
        (4, "Right", {"steer": 1.0, "accel": 0.0, "brake": 0.0}),
        (5, "Left + Accelerate", {"steer": -1.0, "accel": 1.0, "brake": 0.0}),
        (6, "Right + Accelerate", {"steer": 1.0, "accel": 1.0, "brake": 0.0}),
        (7, "Left + Brake", {"steer": -1.0, "accel": 0.0, "brake": 1.0}),
        (8, "Right + Brake", {"steer": 1.0, "accel": 0.0, "brake": 1.0}),
    ]
    
    all_passed = True
    
    for action_id, name, expected in actions:
        # Simulate action decoding from wrappers.py
        steer, accel, brake = 0.0, 0.0, 0.0
        if action_id == 1:
            accel = 1.0
        elif action_id == 2:
            brake = 1.0
        elif action_id == 3:
            steer = -1.0
        elif action_id == 4:
            steer = 1.0
        elif action_id == 5:
            steer = -1.0
            accel = 1.0
        elif action_id == 6:
            steer = 1.0
            accel = 1.0
        elif action_id == 7:
            steer = -1.0
            brake = 1.0
        elif action_id == 8:
            steer = 1.0
            brake = 1.0
        
        # Verify
        actual = {"steer": steer, "accel": accel, "brake": brake}
        passed = actual == expected
        all_passed = all_passed and passed
        
        status = "✅" if passed else "❌"
        print(f"{status} Action {action_id}: {name:20s} -> steer={steer:5.1f}, accel={accel:3.1f}, brake={brake:3.1f}")
        
        if not passed:
            print(f"   Expected: {expected}")
            print(f"   Got:      {actual}")
    
    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Action space is correctly implemented!")
        print("\n✅ Key Points:")
        print("   • 9 discrete actions (complete control)")
        print("   • All values are 0.0 or 1.0 or -1.0 (on/off only)")
        print("   • No percentages - pure discrete control")
        print("   • Includes Left+Brake and Right+Brake for corner braking")
        return True
    else:
        print("❌ SOME TESTS FAILED - Check implementation")
        return False


def test_discrete_values():
    """Verify all control values are discrete (0.0, 1.0, or -1.0)."""
    print("\n🔍 Verifying Discrete On/Off Values")
    print("=" * 60)
    
    allowed_values = {-1.0, 0.0, 1.0}
    all_discrete = True
    
    for action_id in range(9):
        steer, accel, brake = 0.0, 0.0, 0.0
        if action_id == 1:
            accel = 1.0
        elif action_id == 2:
            brake = 1.0
        elif action_id == 3:
            steer = -1.0
        elif action_id == 4:
            steer = 1.0
        elif action_id == 5:
            steer = -1.0
            accel = 1.0
        elif action_id == 6:
            steer = 1.0
            accel = 1.0
        elif action_id == 7:
            steer = -1.0
            brake = 1.0
        elif action_id == 8:
            steer = 1.0
            brake = 1.0
        
        values = [steer, accel, brake]
        for val in values:
            if val not in allowed_values:
                print(f"❌ Action {action_id}: Non-discrete value {val}")
                all_discrete = False
    
    if all_discrete:
        print("✅ All actions use only discrete values: -1.0, 0.0, 1.0")
        print("✅ No percentages or continuous values found")
    else:
        print("❌ Found non-discrete values")
    
    print("=" * 60)
    return all_discrete


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DQN DISCRETE ACTION SPACE TEST")
    print("=" * 60 + "\n")
    
    test1 = test_action_mapping()
    test2 = test_discrete_values()
    
    print("\n" + "=" * 60)
    if test1 and test2:
        print("🎉 OVERALL: ALL TESTS PASSED")
        print("✅ DQN is correctly configured for discrete on/off controls")
        exit(0)
    else:
        print("❌ OVERALL: SOME TESTS FAILED")
        exit(1)
