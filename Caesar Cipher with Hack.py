
#
# # Define the shift value i.e., the number of positions we want to shift from each character.
# # Iterate over each character of the plain text:
# # If the character is upper-case:
# # Calculate the position/index of the character in the 0-25 range.
# # Perform the positive shift using the modulo operation.
# # Find the character at the new position.
# # Replace the current capital letter by this new character.
# # Else, If the character is not upper-case, keep it with no change.
#
#
# # # A python program to illustrate Caesar Cipher Technique
# text = str(input("Please enter the text in upper case :"))
# shift = int(input("Please enter the key:  "))  # defining the shift count
#
# encryption = ""
#
# for c in text:
#
#     # check if character is an uppercase letter
#     if c.isupper():
#
#         # find the position in 0-25
#         c_unicode = ord(c)
#
#         c_index = ord(c) - ord("A")
#
#         # perform the shift
#         new_index = (c_index + shift) % 26
#
#         # convert to new character
#         new_unicode = new_index + ord("A")
#
#         new_character = chr(new_unicode)
#
#         # append to encrypted string
#         encryption = encryption + new_character
#
#     else:
#
#         # since character is not uppercase, leave it as it is
#         encryption += c
#
# print("Plain text:", text)
#
# print("Encrypted text:", encryption)
#
#
# # Decrypting the Cipher text :
#
# encrypted_text = str(input("Please enter the text in upper case :"))
# shift = int(input("Please enter the key:  "))  # defining the shift count
#
#
#
# plain_text = ""
#
# for c in encrypted_text:
#
#     # check if character is an uppercase letter
#     if c.isupper():
#
#         # find the position in 0-25
#         c_unicode = ord(c)
#
#         c_index = ord(c) - ord("A")
#
#         # perform the negative shift
#         new_index = (c_index - shift) % 26
#
#         # convert to new character
#         new_unicode = new_index + ord("A")
#
#         new_character = chr(new_unicode)
#
#         # append to plain string
#         plain_text = plain_text + new_character
#
#     else:
#
#         # since character is not uppercase, leave it as it is
#         plain_text += c
#
# print("Encrypted text:",encrypted_text)
#
# print("Decrypted text:",plain_text)

# Hacking of Caesar Cipher Algorithm
# The technique of trying every possible decryption key is called a brute-force attack

message = 'GIEWIVrGMTLIVrHIQS' #encrypted message
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Loop through every posible key
for key in range(len(LETTERS)):

# It is important to set translated to the blank string so that the
# so that the previous iteration's value for translated is cleared

   translated = ''

# Rest of the program is the same as the orignal Cesar program
#Run the encryption/decryption code on each symbol in the message
   for symbol in message:
      if symbol in LETTERS:
         num = LETTERS.find(symbol) # get the number of the symbol
         num = num - key

        #handle the wrap-around if num is 26 or larger or less than 0
         if num < 0:
            num = num + len(LETTERS)
        # add number's symbol at the end of translated
         translated = translated + LETTERS[num]
      else:
         # just add the symbol without encrypting /decrypting
         translated = translated + symbol
# display the current key being tested ,along with its decryption
print('Hacking key is  #%s: %s' % (key, translated))
#---------------------------------------------------------------------------------------------------------------------------------------------------------
# Using Python Cryptography module
# # Fernet guarantees the at a message encrypted using it cannot be manipulated or read without keys
# import cryptography
# from cryptography.fernet import Fernet
# key = Fernet.generate_key()
# print("The key is : ",key)
# cipher_suite = Fernet(key)
#
# cipher_text = cipher_suite.encrypt(b"Hello")
# print(cipher_text)
# # it uses URL-safe base64- encoded asnd is referred to as Fernet token which is basically a ciphertext
# # cipher_text = cipher_suite.encrypt(b"This example is used to demonstrate cryptography module")
# plain_text = cipher_suite.decrypt(cipher_text)
# print(plain_text)