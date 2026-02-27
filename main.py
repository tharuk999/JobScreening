import kagglehub

# Download latest version
path = kagglehub.dataset_download("mdtalhask/ai-powered-resume-screening-dataset-2025")

print("Path to dataset files:", path)

path="AI_Resume_Screening.csv"
def get_skills(str):
    ind = str.index('"', 1)
    str = str[1:ind]
    return str.replace(" ", "").split(",")

total_skills = []
# set = set()

with open(path, "r") as file:
    header = file.readline()
    while True:
        s = file.readline()
        if not s:
            break
        # print(s)
        skills = get_skills(s)
        dic = {'DeepLearning':0, 'Cybersecurity':0, 'SQL':0, 'Networking':0, 'Pytorch':0, 'EthicalHacking':0, 'React':0, 'TensorFlow':0, 'C++':0, 'Python':0, 'Linux':0, 'MachineLearning':0, 'NLP':0, 'Java':0}
        for skill in skills:
            set.add(skill)
            if skill in dic:
                dic[skill] += 1
        total_skills.append(dic)
        # print(dic)

# print(set)