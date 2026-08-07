"""add lifecycle_uuid to lifecycles

Revision ID: 7c2a9b3d4e51
Revises: 9f4d7c2a1b5e
Create Date: 2026-08-07 02:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = '7c2a9b3d4e51'
down_revision = '9f4d7c2a1b5e'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'lifecycles',
        sa.Column('lifecycle_uuid', sa.String(length=36), nullable=False),
    )
    op.create_index(op.f('ix_lifecycles_lifecycle_uuid'), 'lifecycles', ['lifecycle_uuid'], unique=True)


def downgrade() -> None:
    op.drop_index(op.f('ix_lifecycles_lifecycle_uuid'), table_name='lifecycles')
    op.drop_column('lifecycles', 'lifecycle_uuid')
