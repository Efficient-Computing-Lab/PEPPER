from jtop import jtop

def measure_method():
    with jtop() as jetson:
        # jetson.ok() will provide the proper update frequency
        if jetson.ok():
            # Read tegra stats
            energy_in_mw = jetson.stats.get("Power POM_5V_CPU")
            energy_value = int(energy_in_mw) / 1000
            return energy_value