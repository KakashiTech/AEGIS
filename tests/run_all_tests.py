#!/usr/bin/env python3
"""
Ejecutar todos los tests del BGCE
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess


def run_test(test_file):
    """Ejecutar un archivo de tests"""
    print(f"\n{'='*80}")
    print(f"Ejecutando: {test_file}")
    print('='*80)
    
    result = subprocess.run(
        [sys.executable, test_file],
        capture_output=False,
        text=True
    )
    
    return result.returncode == 0


def main():
    print("=" * 80)
    print("MOTOR BGCE - SUITE DE TESTS")
    print("=" * 80)
    
    tests = [
        'test_mamba3.py',
        'test_lorentz.py',
        'test_vjepa.py',
        'test_vsa.py'
    ]
    
    results = {}
    
    for test in tests:
        test_path = os.path.join(os.path.dirname(__file__), test)
        if os.path.exists(test_path):
            results[test] = run_test(test_path)
        else:
            print(f"No se encontró: {test}")
            results[test] = False
    
    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE TESTS")
    print("=" * 80)
    
    for test, passed in results.items():
        status = "✓ PASÓ" if passed else "✗ FALLÓ"
        print(f"  {test:30s} {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests pasaron")
    
    if passed == total:
        print("\n✓ Todos los tests pasaron!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) fallaron")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
