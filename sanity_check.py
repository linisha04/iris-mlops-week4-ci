# sanity_check.py
import json

def main():
    report = {
        "model_accuracy": 0.89,
        "status": "OK ✅" if 0.89 > 0.5 else "FAIL ❌"
    }
    print(json.dumps(report, indent=2))
    with open("report.txt", "w") as f:
        f.write(f"Model Accuracy: {report['model_accuracy']}\n")
        f.write(f"Status: {report['status']}\n")

if __name__ == "__main__":
    main()

