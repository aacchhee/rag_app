# ops bundle (templates)

Generated: 2026-02-05

This folder contains *sanitized* templates for:
- Nginx site config for /rag/ and /rag2/
- systemd units for rag and rag2
- deployment + update/ingest scripts for rag and rag2
- example root index.html

## Installation paths on the VM (recommended)
- Nginx:   /etc/nginx/sites-available/rag  (symlink to /etc/nginx/sites-enabled/rag)
- systemd: /etc/systemd/system/rag.service
          /etc/systemd/system/rag2.service
- scripts: /usr/local/bin/deploy_rag.sh
          /usr/local/bin/deploy_rag2.sh
          /usr/local/bin/update_and_ingest_rag.sh
          /usr/local/bin/update_and_ingest_rag2.sh
- root index: /var/www/index.html

## Placeholders you must fill
- __PRIMARY_FQDN__ and __ALT_FQDN__ in Nginx config
- Any paths that differ from your VM layout
- Environment files:
    /etc/rag.env
    /etc/rag2.env
  (Do NOT commit real env files with secrets.)
