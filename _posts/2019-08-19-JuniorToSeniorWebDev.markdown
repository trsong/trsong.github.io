---
layout: post
title:  "Advanced Topic 2019 for Junior to Senior Web Dev"
date:   2019-08-19 22:36:32 -0700
categories: Web
---
* This will become a table of contents (this text will be scraped).
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
  - Impossible to decode, and even small change could lead to completely different hashing value

Surprisingly, most of time SSH only use Symmetrical Encryption for communication. As both client and server use Asymmetrical Encryption to exchange a key for prior Symmetrical Encryption before start using it. Such algorithm is called Diffie Helman algorithm.

#### SSH Keys

SSH Steps
1. Diffie-Hellman Key Exchange (Asymmetrical Encryption, safe but time consuming)
2. Arrive at Symmetric Key
3. Use Hashing to make sure of no funny business between client and server
4. Authenticate User

In step 4, the traditional way is to enter a password; However, we cannot avoid brutal force as well as the hassle to memorize/enter the password. A better way is to take advantage of RSA token.

```sh
cd ~/.ssh  # mkdir .ssh  if your local machine don't have one

ssh-keygen
# or
ssh-keygen -t rsa -b 4096 -C "user@domain.com"  # -C is for comment

# as I want to create one for my raspberrypi, I enter ~/.ssh/id_rsa_raspberrypi
# that will generate:
# 1)  ~/.ssh/id_rsa_raspberrypi (private key NEVER share it w/ anyone)
# 2)  ~/.ssh/id_rsa_raspberrypi.pub (public key to share)
cat id_rsa_raspberrypi.pub  # and copy the result to clipboard

ssh pi@RASPBERRY
cd ~/.ssh  # mkdir .ssh  if remote machine don't have one 
vi authorized_keys  # create one if remote don't have one
                    # append content of the public key id_rsa_raspberrypi.pub from clipboad to that file
exit   # exit from remote to local

# for windows WSL only:
eval `ssh-agent -s`

ssh-add ~/.ssh/id_rsa_raspberrypi   # the reason we want to do this is we want ssh use the private key automatically
ssh pi@RASPBERRY  # should not require any password
```

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
 


 