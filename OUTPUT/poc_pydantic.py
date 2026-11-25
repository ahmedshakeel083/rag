from typing import Optional
from pydantic import BaseModel, EmailStr, Field

class Review(BaseModel):
    # product: str
    product: str = "Renewable Energy"
    price: Optional[float] = None
    email: EmailStr
    rating: int = Field(gt=0, lt=6, default=5, description="Rating should be between 1 to 5")

# product_one = { 'product' : "Solar Panels"}
product_one = {
    'product' : "Wind Turbines",
    'price' : 12999.99,
    'email' : "yY5t7@example.com",
    'rating' : 4
}

structured_model = Review(**product_one)

# print(structured_model.email)

print(structured_model.model_dump(), structured_model, structured_model.model_dump_json())
