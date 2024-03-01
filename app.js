import { OpenAI } from 'langchain/llms/openai'
import cors from 'cors';
import { VectorDBQAChain } from 'langchain/chains'
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import * as fs from "fs"
import * as dotenv from 'dotenv'
dotenv.config()

process.env.OPENAI_API_KEY = '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';

const GetAnswer = async (questions, knowledgebase) => {

    // Gabungkan konten dari setiap item menjadi satu teks
    const combinedText = knowledgebase.map(item => item.knowledgebase).join('\r\n\r\n');



    const model = new OpenAI({
        temperature: 0.9,
        // OPENAI_API_KEY: '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'
    })

    // const text = fs.readFileSync("data.txt", "utf8")

    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 })

    // const docs = await textSplitter.createDocuments([text])
    // Buat dokumen dari teks yang telah digabungkan
    const docs = await textSplitter.createDocuments([combinedText]);
    // console.log(docs);

    const vectorStore = await MemoryVectorStore.fromDocuments(docs, new OpenAIEmbeddings())

    const chain = VectorDBQAChain.fromLLM(model, vectorStore)

    // const questions = "saya lupa password."

    const res = await chain.call({
        input_document: docs,
        query: questions
    })

    console.log(questions + " " + res.text);

    return {
        answer: res.text
    }
}


import express from 'express';
const app = express()
app.use(cors());
app.use(express.json())
const port = process.env.PORT || 3030

app.get('/', (req, res) => {
    console.log(res);

    console.log('berhasil berjalan');

    res.send('Hello World!')
})

app.post('/tes', async (req, res) => {
    const resAnswer = await GetAnswer(`${req.body.questions}`, req.body.knowledgebase)
    res.send(resAnswer)
})

app.listen(port, () => {
    console.log(`Server running on port ${port}`)
})

