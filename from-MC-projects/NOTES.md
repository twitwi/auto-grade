
See also ../README.md for details about hacking amc.




# About the annotator

It uses a Python+Flask web server, interfaced using socket.io with the vue.js UI in the browser.
It allows to browse the answers of each student, use OCR (a bad one, together with suggestions and a matching algo) to pre-fill the corresponding answer, allow the user to change it, and to save it.

Currently, some things are strongly hard-coded, this includes:
- the fields (OCR boxes) to extract, and which type they have (see Test3.vue#created)
- the matching algorithm, with the auto-complete list that is a per category list of words (suggestions.js)

To run, in two processes:

~~~
python3 flask-ws.py

# and

cd vue-view
with_node
yarn run dev
~~~

"LOG IT" will append to `all-logs.jstream`, which can be converted to TSV using:

~~~
python3 jstream-to-tsv.py  ../all-logs.jstream

less ../all-logs.jstream.tsv
~~~
