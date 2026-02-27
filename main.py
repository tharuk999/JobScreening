import kagglehub

# Download latest version
path = kagglehub.dataset_download("mdtalhask/ai-powered-resume-screening-dataset-2025")

print("Path to dataset files:", path)

path="AI_Resume_Screening.csv"
def get_skills(str):
    ind = str.index('"', 1)
    str = str[1:ind]
    return str.replace(" ", "").split(",")

set = set()
with open(path, "r") as file:
    header = file.readline()
    while True:
        s = file.readline()
        if not s:
            break
        print(s)
        skills = get_skills(s)
        for skill in skills:
            set.add(skill)

print(set)