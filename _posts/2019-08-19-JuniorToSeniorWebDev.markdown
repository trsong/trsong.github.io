---
layout: post
title:  "Advanced Topic 2019 for Junior to Senior Web Dev"
date:   2019-08-19 22:36:32 -0700
categories: Web
---
{:toc}

### SSH
---
**ssh Usage:**
```sh
ssh {user}@{host}
```

**rsync Usage:**
```sh
rsync -avzh /directory1/file1 /directory2/file2  # sync file locally
rsync -avzh . root@127.0.0.1:~/Desktop           # sync folder remotely
```

#### SSH Encryption

Three techniques used in SSH:
- Symmetrical Encryption
  - Same set of keys to encode and decode data
- Asymmetrical Encryption
  - Two sets of private/public key pairs, only corresponding private key and work with the corresponding public key
- Hashing

Surprisingly, most of time SSH only use Symmetrical Encryption for communication. As both client and server use Asymmetrical Encryption to exchange a key for prior Symmetrical Encryption before start using it. Such algorithm is called Diffie Helman algorithm.

#### SSH Keys

### Performance
#### Network Optimizations
#### Front End Optimizations
#### Back End Optimizations

### React + Redux
#### React
#### Redux
#### Webpack

### Testing
#### Testing
#### Jest
#### React Tests

### Typescript
#### Static Typing
#### Typescript
#### Typescript in React


### SPA vs Server Side
#### Server Side Rendering
#### Client Side Rendering
#### Next.js

### Security
#### Front End Security
#### Back End Security
#### Ethical Hacking


### Docker
#### Containers
#### Docker
#### Docker-Compose

### Redis
#### Databases
#### Redis
#### Redis CLI


### Sessions + JWT
#### Session Authentication
#### Token Authentication
#### Secure Authentication Flow

### AWS
#### AWS
#### AWS Lambda 
#### Serverless


### CI/CD
#### Continuous Integration
#### Continuous Delivery
#### Continuous Deployment
 


 