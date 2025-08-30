#!/usr/bin/env python3
"""
Assignment runner for deep learning practice assignments.
Usage: python run_assignment.py <assignment_name>
"""

import sys
import importlib
import argparse
from pathlib import Path

def run_assignment(assignment_name):
    """Run tests for a specific assignment."""
    assignment_path = Path(f"assignments/{assignment_name}")
    if not assignment_path.exists():
        print(f"Assignment '{assignment_name}' not found!")
        return False
    
    print(f"\n{'='*50}")
    print(f"Running assignment: {assignment_name}")
    print(f"{'='*50}")
    
    try:
        test_module = importlib.import_module(f"assignments.{assignment_name}.test")
        test_module.run_tests()
        print(f"\n✅ Assignment '{assignment_name}' completed successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Assignment '{assignment_name}' failed: {e}")
        return False

def list_assignments():
    """List all available assignments."""
    assignments_dir = Path("assignments")
    if not assignments_dir.exists():
        print("No assignments directory found!")
        return []
    
    assignments = [d.name for d in assignments_dir.iterdir() if d.is_dir()]
    return sorted(assignments)

def main():
    parser = argparse.ArgumentParser(description="Run deep learning practice assignments")
    parser.add_argument("assignment", help="Assignment name or 'all' to run all assignments")
    args = parser.parse_args()
    
    if args.assignment == "all":
        assignments = list_assignments()
        if not assignments:
            print("No assignments found!")
            return
        
        print(f"Found {len(assignments)} assignments: {', '.join(assignments)}")
        passed = 0
        for assignment in assignments:
            if run_assignment(assignment):
                passed += 1
        
        print(f"\n{'='*50}")
        print(f"Summary: {passed}/{len(assignments)} assignments passed")
        print(f"{'='*50}")
    else:
        run_assignment(args.assignment)

if __name__ == "__main__":
    main()