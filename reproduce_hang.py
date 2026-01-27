
import time
import numpy as np
from src.data_utils.synthetic.atomistic_generator import LiquidMetalGenerator, LiquidStructureConfig

def test_liquid_generation():
    L = 300.0
    rho_target = 0.078
    avg_nn_dist = 2.49
    min_dist = 2.1
    
    print(f"Testing LiquidMetalGenerator with L={L}, rho={rho_target}, min_dist={min_dist}")
    print(f"Target atoms: {int(rho_target * L**3)}")
    
    config = LiquidStructureConfig(method='simple')
    rng = np.random.default_rng(42)
    
    start = time.perf_counter()
    
    # We create a subclass or modify expectation because the full generate takes too long
    # We will try to generate a smaller box first to extrapolate, or just run it and see if it feels slow.
    # Actually, let's run a smaller box that should be fast, and a medium one.
    
    # Small box output
    L_small = 50.0
    print(f"\n--- Run 1: Small Box L={L_small} ---")
    gen_small = LiquidMetalGenerator(L_small, rho_target, avg_nn_dist, config, rng, min_pair_dist=min_dist)
    t0 = time.perf_counter()
    atoms_small = gen_small.generate()
    t1 = time.perf_counter()
    print(f"Generated {len(atoms_small)} atoms in {t1-t0:.4f}s")
    rate = len(atoms_small) / (t1-t0)
    print(f"Rate: {rate:.1f} atoms/sec")
    
    # Medium box output
    L_med = 100.0
    print(f"\n--- Run 2: Medium Box L={L_med} ---")
    gen_med = LiquidMetalGenerator(L_med, rho_target, avg_nn_dist, config, rng, min_pair_dist=min_dist)
    t0 = time.perf_counter()
    # This might hang if my hypothesis is true.
    # We'll set a timeout or just letting it run for a bit?
    # Actually, let's just run it. If it takes > 10s for 100^3 (which is 1/27th of 300^3), then 300^3 will take 270s+
    # 100^3 volume is 10^6. Atoms ~ 78,000.
    # 300^3 volume is 27*10^6. Atoms ~ 2,100,000.
    
    atoms_med = gen_med.generate()
    t1 = time.perf_counter()
    print(f"Generated {len(atoms_med)} atoms in {t1-t0:.4f}s")
    rate_med = len(atoms_med) / (t1-t0)
    print(f"Rate: {rate_med:.1f} atoms/sec")
    
    # Extrapolate to 300.0
    n_target_full = int(rho_target * 300.0**3)
    est_time = n_target_full / rate_med
    print(f"\nEstimated time for L=300.0 ({n_target_full} atoms): {est_time:.2f}s ({est_time/60:.2f} min)")

if __name__ == "__main__":
    test_liquid_generation()
