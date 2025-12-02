from langchain.document_loaders import PyPDFLoader

pdf = "data/Reminiscences of a Stock Operator 2008-2.pdf"
docs = PyPDFLoader(pdf).load()
print("pages:", len(docs))
for i, d in enumerate(docs[:20]):  # 前3页示例
    print(f"\n--- page {d.metadata.get('page')} len={len(d.page_content)} ---")
    print(d.page_content[:500])
