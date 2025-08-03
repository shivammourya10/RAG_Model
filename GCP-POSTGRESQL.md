# Setting Up PostgreSQL Database for RAG Model on GCP

This guide explains how to set up a PostgreSQL database on Google Cloud Platform (GCP) for your RAG Model deployment.

## Options for PostgreSQL on GCP

You have three main options for PostgreSQL on GCP:

1. **Cloud SQL for PostgreSQL**: Fully managed PostgreSQL service
2. **Render PostgreSQL**: Continue using your existing Render database
3. **Self-managed PostgreSQL**: Run your own PostgreSQL instance on GCP

## Option 1: Cloud SQL for PostgreSQL (Recommended)

### Step 1: Create a Cloud SQL PostgreSQL Instance

```bash
gcloud sql instances create hackrx-db \
  --database-version=POSTGRES_14 \
  --cpu=1 \
  --memory=3840MB \
  --region=us-central1 \
  --root-password=YOUR_SECURE_PASSWORD
```

### Step 2: Create a Database

```bash
gcloud sql databases create hackrx_db --instance=hackrx-db
```

### Step 3: Create a User

```bash
gcloud sql users create hackrx_user \
  --instance=hackrx-db \
  --password=YOUR_SECURE_PASSWORD
```

### Step 4: Get Connection Information

```bash
gcloud sql instances describe hackrx-db
```

Note the `connectionName` in the output.

### Step 5: Update Your `.env` File

Update your DATABASE_URL in the `.env` file:

```
DATABASE_URL=postgresql://hackrx_user:YOUR_SECURE_PASSWORD@/hackrx_db?host=/cloudsql/YOUR_PROJECT:us-central1:hackrx-db
```

### Step 6: Connect Cloud Run to Cloud SQL

When deploying to Cloud Run, add the Cloud SQL connection:

```bash
gcloud run deploy rag-model \
  --image gcr.io/YOUR_PROJECT/rag-model \
  --platform managed \
  --region us-central1 \
  --add-cloudsql-instances YOUR_PROJECT:us-central1:hackrx-db \
  [other flags...]
```

Update the `deploy-gcp.sh` script to include this flag.

## Option 2: Continue Using Render PostgreSQL

If you want to keep using your existing Render PostgreSQL database:

1. Ensure your Render database allows external connections
2. Configure the IP allow list to include GCP IPs
3. Keep your existing DATABASE_URL in `.env`

This option is simpler but may have higher latency between GCP and Render.

## Option 3: Self-Managed PostgreSQL on GCP

If you need more control, you can run PostgreSQL on a GCP VM:

1. Create a Compute Engine VM instance
2. Install and configure PostgreSQL
3. Set up proper networking and firewall rules
4. Update your DATABASE_URL to point to your VM instance

## Database Migration

If you're moving from an existing database to a new one:

### Export Data from Source

```bash
pg_dump -h your-source-host -U your-source-user -d your-source-db > dump.sql
```

### Import Data to Destination

```bash
psql -h your-destination-host -U your-destination-user -d your-destination-db < dump.sql
```

## Security Best Practices

1. **Use Private IP** when possible
2. **Enable SSL** connections
3. **Use Strong Passwords**
4. **Configure Backup** and point-in-time recovery
5. **Use IAM Authentication** for Cloud SQL

## Monitoring and Maintenance

- Set up alerts for high CPU or memory usage
- Schedule regular backups
- Update PostgreSQL versions when needed
- Monitor slow queries

## Cost Optimization

To reduce costs:

1. Choose appropriate instance size
2. Enable auto-storage scaling
3. Use shared-core instances for dev/test
4. Consider HA requirements carefully (HA doubles cost)
