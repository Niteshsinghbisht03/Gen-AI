from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name:str
    age:Optional[int]=None
    email:EmailStr
    cgpa:float=Field(gt=0,lt=10 ,default=5,description='decimal value represting cgpa')
    
new_student = {'name':'nitish','age':32,'email':'abc@gmail.com','cgpa':9}

student = Student(**new_student)
student_dict = dict(student)
student_json = student.model_dump_json()
print(student_json)
print()