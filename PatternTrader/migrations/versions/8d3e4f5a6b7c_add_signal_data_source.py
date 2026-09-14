"""add data_source to signals

Revision ID: 8d3e4f5a6b7c
Revises: 7c2a9b3d4e51
Create Date: 2026-09-14 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "8d3e4f5a6b7c"
down_revision = "7c2a9b3d4e51"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "signals",
        sa.Column("data_source", sa.String(length=20), nullable=False, server_default="live"),
    )


def downgrade() -> None:
    op.drop_column("signals", "data_source")
