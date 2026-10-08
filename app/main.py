from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Customer, Product, Order, SupportTicket
from app.agent_router import route_question
from app.sql_tool import execute_analytics
from ml.ticket_nlp import analyze_ticket
from ml.priority_predictor import predict_priority
from rag_pipeline import answer_question


app = FastAPI(
    title="Enterprise AI Knowledge & Support Agent",
    description="AI-powered enterprise knowledge and support platform",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "Enterprise AI Agent is running!",
        "status": "healthy",
    }


@app.post("/chat")
def chat(request: ChatRequest):

    route = route_question(request.question)

    # Knowledge questions → RAG
    if route == "KNOWLEDGE":

        result = answer_question(request.question)

        return {
            "question": request.question,
            "route": route,
            "answer": result["answer"],
            "sources": result["sources"],
        }

    # Analytics questions → SQL tool
    elif route == "ANALYTICS":

        result = execute_analytics(
            request.question
        )

        return {
            "question": request.question,
            "route": route,
            "answer": result["answer"],
            "query": result.get("query"),
        }

    # Ticket questions → Ticket analysis
    elif route == "TICKET":

        return {
            "question": request.question,
            "route": route,
            "answer": (
                "This question requires the ticket "
                "analysis tool."
            ),
        }

    # Unsupported questions
    else:

        return {
            "question": request.question,
            "route": "UNSUPPORTED",
            "answer": (
                "I can currently answer company knowledge "
                "questions and selected business analytics "
                "queries."
            ),
        }


@app.get("/customers")
def get_customers(
    db: Session = Depends(get_db)
):

    customers = db.query(Customer).all()

    return [
        {
            "customer_id": customer.customer_id,
            "name": customer.name,
            "email": customer.email,
        }
        for customer in customers
    ]


@app.get("/products")
def get_products(
    db: Session = Depends(get_db)
):

    products = db.query(Product).all()

    return [
        {
            "product_id": product.product_id,
            "name": product.name,
            "category": product.category,
            "price": float(product.price),
            "stock_quantity": product.stock_quantity,
        }
        for product in products
    ]


@app.get("/orders")
def get_orders(
    db: Session = Depends(get_db)
):

    orders = db.query(Order).all()

    return [
        {
            "order_id": order.order_id,
            "customer_id": order.customer_id,
            "product_id": order.product_id,
            "quantity": order.quantity,
            "status": order.status,
        }
        for order in orders
    ]


@app.get("/tickets")
def get_tickets(
    db: Session = Depends(get_db)
):

    tickets = db.query(SupportTicket).all()

    return [
        {
            "ticket_id": ticket.ticket_id,
            "customer_id": ticket.customer_id,
            "subject": ticket.subject,
            "description": ticket.description,
            "status": ticket.status,
            "priority": ticket.priority,
        }
        for ticket in tickets
    ]


@app.get("/tickets/analyze")
def analyze_tickets(
    db: Session = Depends(get_db)
):

    tickets = db.query(SupportTicket).all()

    results = []

    for ticket in tickets:

        nlp_result = analyze_ticket(
            ticket.description
        )

        priority_result = predict_priority(
            ticket_type=nlp_result["category"],
            subject=ticket.subject,
            description=ticket.description,
        )

        results.append(
            {
                "ticket_id": ticket.ticket_id,
                "subject": ticket.subject,
                "description": ticket.description,
                "sentiment": nlp_result["sentiment"],
                "category": nlp_result["category"],
                "predicted_priority": priority_result["priority"],
                "priority_confidence": round(
                    priority_result["confidence"],
                    4
                ),
                "current_priority": ticket.priority,
                "status": ticket.status,
            }
        )

    return results