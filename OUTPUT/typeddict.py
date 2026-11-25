from typing import TypedDict

class RenewableEnergyData(TypedDict):
    country: str
    year: int
    renewable_energy_consumption: float

solar_panel: RenewableEnergyData = {
    "country": "India",
    "year": 2022,
    "renewable_energy_consumption": 1089
}

print(solar_panel)
