import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import fs from "fs";

const urls = [
  "https://shivaboys.edu.tt/",
  "https://shivaboys.edu.tt/contact.html",
  "https://shivaboys.edu.tt/about.html",
  "https://shivaboys.edu.tt/department.html"
];

(async () => {
  const docs = [];

  for (const url of urls) {
    try {
      const loader = new CheerioWebBaseLoader(url);
      const [raw] = await loader.load();
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 600,
        chunkOverlap: 100,
      });
      const chunks = await splitter.splitDocuments([raw]);
      for (const chunk of chunks) {
        docs.push({
          title: raw.metadata.source, 
          content: chunk.pageContent.trim(),
        });
      }
    } catch (err) {
      console.error("⚠️ Failed to fetch:", url, err);
    }
  }

  fs.writeFileSync("data/website_data.json", JSON.stringify(docs, null, 2));
  console.log("✅ Generated website_data.json with", docs.length, "chunks");
})();
