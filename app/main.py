from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Customer, Product, Order, SupportTicket
from ml.ticket_nlp import analyze_ticket
from ml.priority_predictor import predict_priority


app = FastAPI(
    title="Enterprise AI Knowledge & Support Agent",
    description="AI-powered enterprise knowledge and support platform",
    version="0.1.0",
)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "message": "Enterprise AI Agent is running!",
        "status": "healthy",
    }


# =========================================================
# CUSTOMERS
# =========================================================

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


# =========================================================
# PRODUCTS
# =========================================================

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


# =========================================================
# ORDERS
# =========================================================

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


# =========================================================
# SUPPORT TICKETS
# =========================================================

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


# =========================================================
# TICKET ANALYSIS
# =========================================================

@app.get("/tickets/analyze")
def analyze_tickets(
    db: Session = Depends(get_db)
):
    tickets = db.query(SupportTicket).all()

    results = []

    for ticket in tickets:

        # ---------------------------------------------
        # NLP analysis
        # ---------------------------------------------

        nlp_result = analyze_ticket(
            ticket.description
        )

        # ---------------------------------------------
        # ML priority prediction
        # ---------------------------------------------

        priority_result = predict_priority(
            ticket_type=nlp_result["category"],
            subject=ticket.subject,
            description=ticket.description,
        )

        # ---------------------------------------------
        # Combined result
        # ---------------------------------------------

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