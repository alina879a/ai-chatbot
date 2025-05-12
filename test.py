def get_vectorstore(text_chunks):
    print(f"Number of chunks: {len(text_chunks)}")
    if not text_chunks:
        raise ValueError("No text chunks found. Check PDF parsing.")

    embeddings = OpenAIEmbeddings()
    embedded = embeddings.embed_documents(text_chunks)
    if not embedded:
        raise ValueError("Embedding failed. Check API key and text format.")

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
