#!/usr/bin/env python3
"""Run all BGCE test suites."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess


def run_test(test_file):
    print(f"\n{'='*70}")
    print(f"Running: {test_file}")
    print('='*70)
    result = subprocess.run([sys.executable, test_file], capture_output=False, text=True)
    return result.returncode == 0


def main():
    print("=" * 70)
    print("BGCE TEST SUITE")
    print("=" * 70)
    
    test_dir = os.path.dirname(__file__)
    tests = sorted(f for f in os.listdir(test_dir) if f.startswith('test_') and f.endswith('.py'))
    
    results = {}
    for test in tests:
        test_path = os.path.join(test_dir, test)
        results[test] = run_test(test_path)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test:40s} {status}")
    
    total = len(results)
    passed_count = sum(results.values())
    print(f"\nTotal: {passed_count}/{total} tests passed")
    
    if passed_count == total:
        print("\nAll tests passed!")
        return 0
    else:
        print(f"\n{total - passed_count} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
