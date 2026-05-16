// Dados de exemplo que espelham src/data.py do projeto de treinamento.
// Use na inicialização do banco para testes e demos.

import type {
  PersonRecord,
  MedicationRecord,
  LocationRecord,
  AgendaRecord,
  RoutineStep,
} from '../types';

export const SAMPLE_PERSONS: Omit<PersonRecord, 'createdAt' | 'updatedAt'>[] = [
  {
    id: 'person_001',
    name: 'Maria Silva',
    relationship: 'filha',
    bio: 'Filha mais velha. Mora a 10 minutos de distância. Visita toda semana. Trabalha como professora.',
    phoneNumber: '+55 71 99999-0001',
    isCaregiver: true,
  },
  {
    id: 'person_002',
    name: 'Carlos Silva',
    relationship: 'filho',
    bio: 'Filho mais novo. Mora no Rio. Visita nos feriados. É engenheiro.',
    phoneNumber: '+55 21 99999-0002',
    isCaregiver: false,
  },
  {
    id: 'person_003',
    name: 'Ana Costa',
    relationship: 'neta',
    bio: 'Neta de 8 anos, filha da Maria. Adora brincar de cartas.',
    phoneNumber: undefined,
    isCaregiver: false,
  },
  {
    id: 'person_004',
    name: 'Dr. Roberto Almeida',
    relationship: 'médico',
    bio: 'Médico neurologista. Consultas toda primeira segunda-feira do mês.',
    phoneNumber: '+55 71 3333-0004',
    isCaregiver: true,
  },
  {
    id: 'person_005',
    name: 'Cida Oliveira',
    relationship: 'cuidadora',
    bio: 'Cuidadora profissional. Vem de segunda a sexta, das 8h às 17h.',
    phoneNumber: '+55 71 99999-0005',
    isCaregiver: true,
  },
];

export const SAMPLE_MEDICATIONS: Omit<MedicationRecord, 'id'>[] = [
  {
    name: 'Donepezila 10mg',
    dosage: '1 comprimido',
    timeOfDay: 'evening',
    description: 'Comprimido pequeno branco redondo',
    instructions: 'Tomar após o jantar com um copo de água',
    isActive: true,
  },
  {
    name: 'Losartana 50mg',
    dosage: '1 comprimido',
    timeOfDay: 'morning',
    description: 'Comprimido branco oblongo com risco no meio',
    instructions: 'Tomar em jejum pela manhã',
    isActive: true,
  },
  {
    name: 'Atorvastatina 20mg',
    dosage: '1 comprimido',
    timeOfDay: 'night',
    description: 'Comprimido oval branco pequeno',
    instructions: 'Tomar antes de dormir',
    isActive: true,
  },
  {
    name: 'Vitamina D 2000UI',
    dosage: '1 cápsula',
    timeOfDay: 'morning',
    description: 'Cápsula gelatinosa amarela pequena',
    instructions: 'Tomar junto com o café da manhã',
    isActive: true,
  },
];

export const SAMPLE_LOCATIONS: Omit<LocationRecord, 'id'>[] = [
  {
    name: 'Cozinha',
    description: 'A cozinha tem fogão, geladeira e pia. Fica no fundo da casa.',
    features: ['fogão', 'geladeira', 'pia', 'microondas', 'azulejo'],
    navigationHint: 'A cozinha fica no fundo da casa, ao fim do corredor.',
  },
  {
    name: 'Quarto',
    description: 'Seu quarto tem a cama de casal, o armário marrom e a janela para o jardim.',
    features: ['cama', 'armário', 'travesseiro', 'janela', 'abajur'],
    navigationHint: 'Seu quarto é o primeiro à direita no corredor.',
  },
  {
    name: 'Banheiro',
    description: 'O banheiro principal tem vaso sanitário, pia e chuveiro.',
    features: ['vaso sanitário', 'pia', 'chuveiro', 'espelho', 'toalha'],
    navigationHint: 'O banheiro fica ao lado do seu quarto, à direita.',
  },
  {
    name: 'Sala',
    description: 'A sala tem o sofá, a televisão e as fotos da família na parede.',
    features: ['sofá', 'televisão', 'tapete', 'fotos', 'janela'],
    navigationHint: 'A sala fica na entrada da casa, antes do corredor.',
  },
  {
    name: 'Varanda',
    description: 'A varanda tem cadeiras e o jardim. Bom lugar para tomar sol pela manhã.',
    features: ['cadeiras', 'plantas', 'jardim', 'céu', 'grade'],
    navigationHint: 'A varanda fica pela porta da frente ou pela sala.',
  },
];

export const SAMPLE_AGENDA: Omit<AgendaRecord, 'id'>[] = [
  {
    title: 'Consulta Dr. Roberto',
    description: 'Consulta de rotina com o neurologista no Hospital São Rafael.',
    dayOfWeek: 1, // segunda
    time: '10:00',
    location: 'Hospital São Rafael',
    isRecurring: false,
  },
  {
    title: 'Visita da Maria',
    description: 'Sua filha Maria vem te visitar para almoçar juntos.',
    dayOfWeek: 0, // domingo
    time: '12:00',
    location: 'Casa',
    isRecurring: true,
  },
  {
    title: 'Fisioterapia',
    description: 'Sessão de fisioterapia para exercícios de mobilidade.',
    dayOfWeek: 3, // quarta
    time: '14:00',
    location: 'Clínica Bem Estar',
    isRecurring: true,
  },
];

export const SAMPLE_ROUTINE: Omit<RoutineStep, 'id'>[] = [
  { timeOfDay: 'morning', order: 1, title: 'Acordar', description: 'Hora de se levantar devagar, primeiro sente na cama antes de se levantar.' },
  { timeOfDay: 'morning', order: 2, title: 'Tomar banho', description: 'Banho morno no banheiro ao lado do seu quarto.' },
  { timeOfDay: 'morning', order: 3, title: 'Café da manhã', description: 'Café com leite e pão na cozinha. Lembre dos remédios da manhã.' },
  { timeOfDay: 'morning', order: 4, title: 'Remédios', description: 'Losartana e Vitamina D depois do café.' },
  { timeOfDay: 'afternoon', order: 1, title: 'Almoço', description: 'Almoço na cozinha. A Cida prepara a comida.' },
  { timeOfDay: 'afternoon', order: 2, title: 'Descanso', description: 'Descanse um pouco no quarto ou no sofá da sala.' },
  { timeOfDay: 'evening', order: 1, title: 'Jantar', description: 'Jantar leve. Lembre da Donepezila depois de jantar.' },
  { timeOfDay: 'night', order: 1, title: 'Remédio da noite', description: 'Atorvastatina antes de dormir com um copo de água.' },
  { timeOfDay: 'night', order: 2, title: 'Dormir', description: 'Hora de descansar. Boa noite!' },
];
