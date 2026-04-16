from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pinecone import Pinecone
import requests, io, torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP model loaded.")

PINECONE_API_KEY = "TU_PINECONE_API_KEY_AQUI"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("productos-catalogo")

def get_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features[0].tolist()

# ——— INDEX a product by URL ———
@app.post("/index-product")
async def index_product(product_id: str, image_url: str):
    try:
        response = requests.get(image_url, timeout=10)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        embedding = get_embedding(image)
        index.upsert(vectors=[{"id": product_id, "values": embedding}])
        return {"status": "indexed", "product_id": product_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ——— SEARCH by uploaded image ———
@app.post("/search")
async def search_similar(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        embedding = get_embedding(image)
        results = index.query(vector=embedding, top_k=6, include_metadata=False)
        ids = [match["id"] for match in results["matches"]]
        scores = [round(match["score"], 3) for match in results["matches"]]
        return {"similar_product_ids": ids, "scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ——— HEALTH CHECK ———
@app.get("/health")
async def health():
    return {"status": "ok"}
