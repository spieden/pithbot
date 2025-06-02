##
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

from vibes import load_text_files, parse_line, combine_slices

model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True)

##
spalding = [
    {
        "title": f["basename"],
        "slices": combine_slices(map(parse_line, f["content"].split("\n")))
    }
    for f in load_text_files("spalding")
]

##

print(spalding[1]["slices"])

##

def make_id(title, start, end):
    return f"{title}-{start}-{end}"


data = [
    {
        "id": make_id(mono["title"], slice["start_index"], slice["end_index"]),
        "title": mono["title"],
        "start": slice["start_index"],
        "end": slice["end_index"],
        "text": slice["text"],
        "text_embedding": vector
    }
    for mono in spalding
    for slice, vector in zip(
        mono["slices"],
        model.encode(list(map(lambda s: s["text"], mono["slices"])))
    )
]

##
client = MilvusClient("http://verse.hansens.haus:19530", token="root:Milvus")

# client.create_database(db_name="sandbox")

##
if client.has_collection(collection_name="spalding"):
    client.drop_collection(collection_name="spalding")

schema = client.create_schema()
schema.add_field("id", datatype=DataType.VARCHAR, max_length=512, is_primary=True)
schema.add_field("title", datatype=DataType.VARCHAR, max_length=512)
schema.add_field("start", datatype=DataType.FLOAT)
schema.add_field("end", datatype=DataType.FLOAT)
schema.add_field("text", datatype=DataType.VARCHAR, max_length=512)
schema.add_field("text_embedding", datatype=DataType.FLOAT_VECTOR, dim=1024)

client.create_collection(
    collection_name="spalding",
    schema=schema,
)

##

client.insert(collection_name="spalding", data=data)

##
index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    field_name="text_embedding",
    metric_type="COSINE",
    index_type="IVF_FLAT",
    index_name="text_embedding_cosine",
    params={"nlist": 128}
)

client.create_index(
    collection_name="spalding",
    index_params=index_params,
    sync=True
)

client.load_collection("spalding")

##
def query(text):
    return client.search(
        collection_name="spalding",
        data=[model.encode(text)],
        limit=10,
        output_fields=["text", "title", "start", "end"]
    )

def print_result(q):
    print("\n".join([d["entity"]["text"] for d in query(q)[0]]))

print_result("skipping ritual daily libations") # -> "what's a day without cocktail hour?"

