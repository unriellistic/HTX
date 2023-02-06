import json
import os
cwd = os.chdir("C:\\Users\\alpha\\OneDrive\\Desktop\\Life\\NTU\\Internship\\HTX\\json test files\\LAG + Threat")

for filename in os.listdir(cwd):
    print(filename)
    if filename.endswith(".json"):
        with open(f"{filename}", 'r+') as openfile:
            # Reading from json file
            try:
                json_object = json.loads(openfile.read())
                print("Old Json data:", json_object)
                openfile.seek(0)  # rewind
                json.dump(json_object, openfile, indent=4, sort_keys=True)
                print("New Json data:", json_object)
                openfile.truncate()
            except:
                print(filename+" is empty")





