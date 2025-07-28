"""
Cold Start Performance Monitor
=============================

Monitors and logs cold start performance improvements.
"""

import time
import logging
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)

class ColdStartMonitor:
    """Monitor cold start performance and optimizations."""
    
    def __init__(self):
        self.start_time = time.time()
        self.checkpoints = {}
        self.optimizations = []
    
    def checkpoint(self, name: str):
        """Record a performance checkpoint."""
        elapsed = time.time() - self.start_time
        self.checkpoints[name] = elapsed
        logger.info(f"⏱️ Checkpoint '{name}': {elapsed:.2f}s")
    
    def log_optimization(self, name: str, time_saved: float):
        """Log an optimization and time saved."""
        self.optimizations.append({
            'name': name,
            'time_saved': time_saved,
            'timestamp': time.time() - self.start_time
        })
        logger.info(f"🚀 Optimization '{name}': saved {time_saved:.2f}s")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_time = time.time() - self.start_time
        total_saved = sum(opt['time_saved'] for opt in self.optimizations)
        
        return {
            'total_startup_time': total_time,
            'checkpoints': self.checkpoints,
            'optimizations': self.optimizations,
            'total_time_saved': total_saved,
            'optimizations_enabled': self._get_enabled_optimizations()
        }
    
    def _get_enabled_optimizations(self) -> Dict[str, bool]:
        """Get status of enabled optimizations."""
        return {
            'cold_start_optimization': os.getenv('RAG_COLD_START_OPTIMIZATION', 'false').lower() == 'true',
            'preload_models': os.getenv('RAG_PRELOAD_MODELS', 'false').lower() == 'true',
            'concurrent_init': os.getenv('RAG_CONCURRENT_INIT', 'false').lower() == 'true',
            'render_mode': os.getenv('RAG_RENDER_MODE', 'false').lower() == 'true',
            'fast_startup': os.getenv('RAG_FAST_STARTUP', 'false').lower() == 'true',
            'skip_pinecone_verification': os.getenv('RAG_SKIP_PINECONE_VERIFICATION', 'false').lower() == 'true'
        }
    
    def print_summary(self):
        """Print detailed performance summary."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("🚀 COLD START PERFORMANCE SUMMARY")
        print("="*60)
        print(f"⏱️ Total Startup Time: {summary['total_startup_time']:.2f}s")
        print(f"💾 Time Saved by Optimizations: {summary['total_time_saved']:.2f}s")
        
        print(f"\n📊 Performance Checkpoints:")
        for name, elapsed in summary['checkpoints'].items():
            print(f"   {name}: {elapsed:.2f}s")
        
        print(f"\n🔧 Active Optimizations:")
        for opt, enabled in summary['optimizations_enabled'].items():
            status = "✅" if enabled else "❌"
            print(f"   {status} {opt}")
        
        if summary['optimizations']:
            print(f"\n⚡ Applied Optimizations:")
            for opt in summary['optimizations']:
                print(f"   🚀 {opt['name']}: saved {opt['time_saved']:.2f}s")
        
        print("="*60)

# Global monitor instance
cold_start_monitor = ColdStartMonitor()
