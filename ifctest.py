import ifcopenshell
import ifcopenshell.geom

settings = ifcopenshell.geom.settings()

f = ifcopenshell.open(r"C:\Users\samue\_dev\datasets\ifc\tvc\SRL-WPD-TVC-UTU8-MOD-CCP-BUW-000001.ifc")
# Assuming 'f' is your ifcopenshell.file instance and 'geom_settings' is configured
it = ifcopenshell.geom.iterator(settings, f, 1)  # Start with single-threaded

# Check file validity first
if not f.good():
    print("File parsing failed:", ifcopenshell.get_log())
    raise ValueError("Invalid IFC file")

print("Attempting iterator init...")
try:
    success = it.initialize()
    if not success:
        print("Iterator init returned False. Logs:", ifcopenshell.get_log())
        raise RuntimeError("Iterator init failed - check IFC validity")
except RuntimeError as e:
    print(f"Explicit error: {e}")
    print("Full logs:", ifcopenshell.get_log())  # Often shows token errors
    raise

# If init succeeds, proceed to loop
while it.next():
    print("Processing shape...")
    shape = it.get()
    # Process shape...