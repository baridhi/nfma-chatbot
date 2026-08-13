# [NFMA chatbot](https://baridhi.github.io/nfma-chatbot)

The webpage hosts an online AI chatbot for the NFMA. This webpage is powered by a local RAG pipeline ("privateGPT") described in more detail [here](https://github.com/baridhi/privateGPT_cfa). Using a Cloudflare tunnel, the JS widget relays the questions and responses back and forth between web users (at NFMA) and the local RAG engine. This linkage/tunnel currntly has to be refreshed with every instance of the RAG session.

## Deploy

1. Create GitHub repository:
   nfma-chatbot.github.io

2. Upload all files from this package.

3. Enable GitHub Pages:
   Settings -> Pages -> Deploy from branch -> main

4. Visit:
   [https://nfma-chatbot.github.io](https://baridhi.github.io/nfma-chatbot)

## Adding new sections

Add another navigation item:

<li><a href="#resources">Resources</a></li>

and create:

<section id="resources">...</section>
