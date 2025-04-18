from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int
    
new_person: Person = {'name':'Nitesh','age':52}

print(new_person)