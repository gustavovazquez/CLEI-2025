import subprocess

def run():
    print("Iniciando Benchmark Completo (100 repeticiones por experimento)...")
    command = [
        "python", "benchmark_parallel.py",
        "--n_rep", "100",
        "--output", "final_benchmark_results.csv"
    ]
    
    try:
        subprocess.run(command, check=True)
        print("\n" + "="*60)
        print("¡Benchmark finalizado con éxito!")
        print("Los resultados se han guardado en: final_benchmark_results.csv")
        print("="*60)
    except subprocess.CalledProcessError as e:
        print(f"\nError durante la ejecución: {e}")
    except KeyboardInterrupt:
        print("\nEjecución cancelada por el usuario.")

if __name__ == "__main__":
    run()
