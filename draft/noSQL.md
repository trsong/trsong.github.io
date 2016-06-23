## Up and Running with NoSQL Database
Link: https://www.lynda.com/Cassandra-tutorials/Welcome/111598/117563-4.html

### What is NoSQL
- SQL for relational database: table-based, more like a spreadsheet
- NoSQL, records store in Row; column represent Fields in row
- Nested values are common in NoSQL
- Fields not standardlized between records

###Tradeoffs and Limitations
- Not solving web scalability issues
- Offer flexibility unavailable to relational database

### document store
- stored in a structured format (XML, JSON, ...)
- organized into "collections"  or "database"
- individual document can have unique structures
- each doc has a specific key, used for query
- possible to query a doc by fields

### key-value store
- query value using key
- drawback: can't query other than the key
- solution: some key-value store allow define more than one key
- sometimes used alongside relational db for caching

### BigTable/tabular
- named after Google's implementation "BigTable"
- each row can have different numbers of columns
- designed for large number of columns
- rows are typically versioned

### graph
- designed for data best represents as interconnected nodes
eg. data of a series of road intersections

### object database
- tightly integrated w/ OOP
- acts as persistent layer: store object directly
- can link object through pointers

### Benefits of NoSQL
- EZ web app with customized fields
- Use as a caching layer
- Store binary files
- serve full web applications

### How to install CouchDB and Futon on ubuntu 14.04
Link: https://www.digitalocean.com/community/tutorials/how-to-install-couchdb-and-futon-on-ubuntu-14-04

#### Setup SSH tunnel from local computer to remote server
You can use the following command, run from your local computer, to set up the tunnel:

```bash
ssh -L5984:127.0.0.1:5984 sammy@your_server_ip
```

Note: Remember to replace sammy with your username and your_server_ip with the IP address of your Droplet.

While the connection is open, you can access Futon from your favorite web browser, using port 5984. Visit this URL to display the helpful Futon page:
http://localhost:5984/_utils



