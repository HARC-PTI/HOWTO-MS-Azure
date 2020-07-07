# Encrypted Image Classification

Microsoft has released their Homomorphic Encryption library [SEAL](https://github.com/Microsoft/SEAL) as [python package](https://pypi.org/project/encrypted-inference).
This allows us to perform classification on encrypted data without the need to ever decrypt it. Alongside the release of the python package Microsoft realeased a notebook detailing how this can be used on Azure: 

https://github.com/Azure/MachineLearningNotebooks/blob/master/tutorials/image-classification-mnist-data/img-classification-part3-deploy-encrypted.ipynb


# Homomorphic Encryption

What Homomorphic Encryption is and how it works goes far beyond this How-To, but this should give you a rough idea:

[HomomorphicEncryption.org](https://homomorphicencryption.org/):
> Fully homomorphic encryption, or simply homomorphic encryption, refers to a class of encryption methods envisioned by Rivest, Adleman, and Dertouzos already in 1978, and first constructed by Craig Gentry in 2009. Homomorphic encryption differs from typical encryption methods in that it allows computation to be performed directly on encrypted data without requiring access to a secret key. The result of such a computation remains in encrypted form, and can at a later point be revealed by the owner of the secret key.
