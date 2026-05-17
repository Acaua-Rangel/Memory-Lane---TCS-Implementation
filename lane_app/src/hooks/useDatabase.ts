import { useState, useEffect } from 'react';
import { LaneDatabase } from '../database/LaneDatabase';

let _dbReady = false;
let _dbPromise: Promise<LaneDatabase> | null = null;

function ensureDatabase(): Promise<LaneDatabase> {
  if (_dbReady) return Promise.resolve(LaneDatabase.getInstance());
  if (_dbPromise) return _dbPromise;

  _dbPromise = (async () => {
    const db = LaneDatabase.getInstance();
    if (!db.isOpen()) {
      await db.open();
      await db.seedIfEmpty();
    }
    _dbReady = true;
    return db;
  })();

  return _dbPromise;
}

export function useDatabase() {
  const [db, setDb] = useState<LaneDatabase | null>(_dbReady ? LaneDatabase.getInstance() : null);
  const [isReady, setIsReady] = useState(_dbReady);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (_dbReady) return;
    ensureDatabase()
      .then((database) => {
        setDb(database);
        setIsReady(true);
      })
      .catch((err) => {
        console.error('[useDatabase]', err);
        setError(String(err));
      });
  }, []);

  return { db, isReady, error };
}
