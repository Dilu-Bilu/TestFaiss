import dspy
from dspy.retrieve import faiss_rm

document_chunks = [
    "The superbowl this year was played between the San Francisco 49ers and the Kanasas City Chiefs",
    "Pop corn is often served in a bowl",
    "The Rice Bowl is a Chinese Restaurant located in the city of Tucson, Arizona",
    "Mars is the fourth planet in the Solar System",
    "An aquarium is a place where children can learn about marine life",
    "The capital of the United States is Washington, D.C",
    "Rock and Roll musicians are honored by being inducted in the Rock and Roll Hall of Fame",
    "Music albums were published on Long Play Records in the 70s and 80s",
    "Sichuan cuisine is a spicy cuisine from central China",
    "The interest rates for mortgages is considered to be very high in 2024",
]

frm = faiss_rm.FaissRM(document_chunks)
turbo = dspy.OpenAI(model="gpt-3.5-turbo")
dspy.settings.configure(lm=turbo, rm=frm)
print(frm(["I am in the mood for Chinese food"]))
