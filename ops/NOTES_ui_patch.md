# UI base-path patching

The shipped UI in /var/www/rag-ui and /var/www/rag2-ui contains hardcoded endpoints:
  - fetch("/ask", ...) and a display line showing Backend: /ask.
For prefix deployments behind Nginx, patch to:
  - /rag/ask for rag
  - /rag2/ask for rag2

Example commands:

  sudo cp /var/www/rag-ui/index.html /var/www/rag-ui/index.html.bak
  sudo sed -i \
    -e 's|Backend: <code>/ask</code> — UI: <code>/rag/</code>.|Backend: <code>/rag/ask</code> — UI: <code>/rag/</code>.|g' \
    -e 's|fetch("/ask"|fetch("/rag/ask"|g' \
    /var/www/rag-ui/index.html

  sudo cp /var/www/rag2-ui/index.html /var/www/rag2-ui/index.html.bak
  sudo sed -i \
    -e 's|Backend: <code>/ask</code> — UI: <code>/rag/</code>.|Backend: <code>/rag2/ask</code> — UI: <code>/rag2/</code>.|g' \
    -e 's|fetch("/ask"|fetch("/rag2/ask"|g' \
    /var/www/rag2-ui/index.html
