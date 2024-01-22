import { OpenAI } from 'langchain/llms/openai'
import { VectorDBQAChain } from 'langchain/chains'
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import * as fs from "fs"
import * as dotenv from 'dotenv'
dotenv.config()

process.env.OPENAI_API_KEY = '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';

const GetAnswer = async (questions, knowledgebase) => {

    // const jsonData = [
    //     {
    //         "id": 1,
    //         "knowledgebase": "Hai, saya adalah BOT Helpdesk yang akan membantu kamu jika mengalami permasalahan yang perlu ditangani oleh TIM IT."
    //     },
    //     {
    //         "id": 2,
    //         "knowledgebase": "Tahapan untuk mengatasi lupa password pada website Wzone: 1. Mengunjungi wzone 2. Klik tombol lupa password 3. Mengisi email. 4. Link reset password akan dikirimkan ke email anda. 5. Cek email ada secara berkala. 6. Klik link yang dikirimkan via email. 7. Masukan password baru untuk mereset password."
    //     },
    //     {
    //         "id": 3,
    //         "knowledgebase": "jika pertanyaan bukan merujuk untuk nanya, jawablah dengan AI, dan jika pertanyaan berada diluar konteks dari data yang diberikan, berikan pesan 'Pertanyaan anda diluar konteks HELPDESK, silahkan mengajukan tiket jika dirasa perlu dijawab oleh TIM IT' serta berikan juga tombol link html yang mengarah pada link https://sla/create-ticket"
    //     },
    //     {
    //         "id": 4,
    //         "knowledgebase": "bersikaplah dengan ramah."
    //     }
    // ];

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
app.use(express.json())
const port = process.env.PORT || 3030

app.get('/', (req, res) => {
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

