"""add trades and equity_curve columns to backtests

Revision ID: 9f4d7c2a1b5e
Revises: 292ed36c3e49
Create Date: 2026-08-07 01:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = '9f4d7c2a1b5e'
down_revision = '292ed36c3e49'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('backtests', sa.Column('trades', sa.JSON(), nullable=True))
    op.add_column('backtests', sa.Column('equity_curve', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('backtests', 'equity_curve')
    op.drop_column('backtests', 'trades')
